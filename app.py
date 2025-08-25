# app.py - Streamlit app using streamlit-webrtc for real-time webcam prediction

import streamlit as st
import numpy as np
import cv2
import os
import mediapipe as mp
import asyncio
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from ccheck import (
    mediapipe_detection,
    draw_styled_landmarks,
    extract_keypoints,
    create_data_folders,
    load_and_prepare_data,
    build_and_train_model
)

# --- Ensure proper asyncio event loop setup for WebRTC ---
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


st.set_page_config(layout="wide", page_title="Indian Sign Language Recognition")
st.title("🤟 Indian Sign Language (ISL) Word Recognition")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")
DATA_PATH = st.sidebar.text_input("Data Path", "MP_Data")
MODEL_PATH = st.sidebar.text_input("Model Path", "action.h5")
actions_input = st.sidebar.text_input("Actions (comma-separated)", "Hello,Thanks,ILoveYou,Yes,No")
actions = [action.strip() for action in actions_input.split(',') if action.strip()]
no_sequences = st.sidebar.slider("Number of Sequences per Action", 1, 100, 30)
sequence_length = st.sidebar.slider("Frames per Sequence", 10, 100, 30)
epochs = st.sidebar.slider("Training Epochs", 10, 500, 100)

if 'collecting' not in st.session_state:
    st.session_state.collecting = False
if 'action_to_collect' not in st.session_state:
    st.session_state.action_to_collect = None
if 'model' not in st.session_state:
    st.session_state.model = None

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
}

mp_holistic = mp.solutions.holistic

# --- Video Processor for Data Collection ---
class DataCollectorProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.action = None
        self.path = None
        self.sequence = 0
        self.frame_num = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        try:
            if self.action is None:
                self.action = st.session_state.get('action_to_collect')
                if self.action:
                    self.path = os.path.join(DATA_PATH, self.action)
                else:
                    cv2.putText(image, "ERROR: Action not set.", (15, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    return image

            is_collecting = st.session_state.get('collecting', False)

            if is_collecting and self.sequence < no_sequences:
                _, results = mediapipe_detection(image, self.holistic)
                draw_styled_landmarks(image, results)
                cv2.putText(image, f'Collecting: {self.action} - Seq {self.sequence + 1}, Frame {self.frame_num + 1}',
                            (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                keypoints = extract_keypoints(results)
                seq_path = os.path.join(self.path, str(self.sequence))
                os.makedirs(seq_path, exist_ok=True)
                np.save(os.path.join(seq_path, str(self.frame_num)), keypoints)

                self.frame_num += 1
                if self.frame_num >= sequence_length:
                    self.frame_num = 0
                    self.sequence += 1
            else:
                _, results = mediapipe_detection(image, self.holistic)
                draw_styled_landmarks(image, results)
                if self.sequence >= no_sequences:
                    cv2.putText(image, f'Collection for {self.action} complete!', (15, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, 'Ready to collect.', (15, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        except Exception as e:
            st.error(f"Error in data collector: {e}")
            cv2.putText(image, "PROCESSING ERROR", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return image


# --- Video Processor for Real-Time Prediction ---
class PredictorProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence = []
        self.sentence = []
        self.threshold = 0.8

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        try:
            if st.session_state.model:
                _, results = mediapipe_detection(image, self.holistic)
                draw_styled_landmarks(image, results)

                keypoints = extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-sequence_length:]

                if len(self.sequence) == sequence_length:
                    res = st.session_state.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    prediction = actions[np.argmax(res)]
                    if res[np.argmax(res)] > self.threshold and (not self.sentence or prediction != self.sentence[-1]):
                        self.sentence.append(prediction)

                if len(self.sentence) > 5:
                    self.sentence = self.sentence[-5:]

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(self.sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Model not loaded.", (15, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        except Exception as e:
            st.error(f"Error in predictor: {e}")
            cv2.putText(image, "PREDICTION ERROR", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return image


# --- Main layout ---
tab1, tab2, tab3 = st.tabs(["1️⃣ Data Collection", "2️⃣ Model Training", "3️⃣ Real-Time Prediction"])

with tab1:
    st.header("Step 1: Collect Keypoint Data")
    st.warning("🚨 For smooth video, check camera permissions and stable internet.")
    selected_action = st.selectbox("Choose an action", actions, key="action_select")

    if st.button("Start/Stop Collection"):
        st.session_state.collecting = not st.session_state.collecting
        if st.session_state.collecting:
            st.session_state.action_to_collect = selected_action
            create_data_folders(DATA_PATH, actions, no_sequences)
            st.warning(f"Starting collection for **{selected_action}**.")
        else:
            st.info("Collection stopped.")

    if st.session_state.collecting:
        webrtc_streamer(
            key="collector",
            video_processor_factory=DataCollectorProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

with tab2:
    st.header("Step 2: Train the LSTM Model")
    if st.button("Train Model"):
        if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
            st.error("Data folder is empty. Please collect data first.")
        else:
            with st.spinner("Training model... This may take time."):
                try:
                    X, y = load_and_prepare_data(DATA_PATH, actions, no_sequences, sequence_length)
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
                    model = build_and_train_model(X_train, y_train, X_test, y_test, epochs, actions)
                    model.save(MODEL_PATH)
                    st.session_state.model = model
                    st.success(f"✅ Model trained and saved to `{MODEL_PATH}`.")
                except Exception as e:
                    st.error(f"Training error: {e}")

with tab3:
    st.header("Step 3: Real-Time Prediction")
    st.warning("🚨 Ensure camera access and stable connection.")

    if st.button("Load Model"):
        if os.path.exists(MODEL_PATH):
            st.session_state.model = load_model(MODEL_PATH)
            st.success("Model loaded successfully!")
        else:
            st.error(f"Model not found at `{MODEL_PATH}`. Train first.")

    run_pred = st.checkbox("Start / Stop Prediction")
    if run_pred:
        if st.session_state.model:
            webrtc_streamer(
                key="predictor",
                video_processor_factory=PredictorProcessor,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        else:
            st.warning("Load a model before starting prediction.")

st.markdown("---")
st.markdown("Made with ❤️ using MediaPipe, TensorFlow, and Streamlit")
