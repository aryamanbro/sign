import streamlit as st
import numpy as np
import cv2
import os
from streamlit_webrtc import VideoTransformerBase
from tensorflow.keras.models import load_model
from datetime import datetime
from ccheck import (
    create_data_folders,
    load_data,
    train_model,
    real_time_prediction,
    KeypointCollector
)



st.set_page_config(layout="wide")
st.title("🤟 Indian Sign Language Recognition")

# Sidebar inputs
st.sidebar.header("Settings")
actions_input = st.sidebar.text_input("Actions (comma-separated)", value="hello,thanks,iloveyou")
actions = [a.strip() for a in actions_input.split(",") if a.strip() != ""]

no_sequences = st.sidebar.slider("Sequences per action", 5, 100, 30)
sequence_length = st.sidebar.slider("Frames per sequence", 10, 60, 30)
epochs = st.sidebar.slider("Epochs", 10, 300, 200)
data_path = st.sidebar.text_input("Data Path", value="MP_Data")
model_path = st.sidebar.text_input("Model Path", value="action.keras")

# Main buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📹 Collect Data"):
        create_data_folders(data_path, actions, no_sequences)
        collector = KeypointCollector(data_path, actions, no_sequences, sequence_length)
        collector.start_collection(actions.index(selected_action))

        webrtc_streamer(
            key="collector",
            video_transformer_factory=lambda: collector,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )

with col2:
    if st.button("🧠 Train Model"):
        st.info("Loading data and training model...")
        X, y, label_map = load_data(data_path, np.array(actions), no_sequences, sequence_length)
        model, X_test, y_test = train_model(X, y, epochs, model_path=model_path)
        st.success("✅ Model training complete and saved.")

with col3:
    run_prediction = st.checkbox("🎯 Run Real-Time Prediction")
    
    if run_prediction:
        if not os.path.exists(model_path):
            st.error("Trained model not found. Please train the model first.")
        else:
            st.warning("Opening webcam... Close window or stop the stream to exit prediction.")
            model = load_model(model_path)
            real_time_prediction(model, np.array(actions), sequence_length)
            st.success("Prediction session ended.")

st.markdown("---")
st.markdown("Made with ❤ using MediaPipe, TensorFlow, and Streamlit")