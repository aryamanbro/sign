import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model and action labels
model = load_model("action.h5")  # Replace with your model path
actions = np.array(["hello", "thanks", "yes", "no"])  # Replace with your action labels
sequence = []
sequence_length = 30
threshold = 0.8

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.sequence = []
        self.last_action = ""
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5,
                                             min_tracking_confidence=0.5)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image, results = mediapipe_detection(image, self.holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-sequence_length:]

        if len(self.sequence) == sequence_length:
            res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
            if res[np.argmax(res)] > threshold:
                self.last_action = actions[np.argmax(res)]

        cv2.rectangle(image, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(image, self.last_action, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

<<<<<<< HEAD
# ---------------------------
# 4. Model Training Function
# ---------------------------

def train_model(X, y, epochs, log_dir="Logs", model_path="action.keras", mapping_path="label_mapping.npy"):
    """Builds, trains, and saves an LSTM model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    
    # Build the model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()
    
    # Setup TensorBoard callback for logging
    tb_callback = TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, epochs=epochs, callbacks=[tb_callback])
    
    # Save the model and mapping (mapping can be saved separately if needed)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model, X_test, y_test

# ---------------------------
# 5. Probability Visualization Function
# ---------------------------

def prob_viz(res, actions, input_frame, colors):
    """Visualize probabilities as rectangles on the input frame."""
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# ---------------------------
# 6. Real-Time Prediction Function
# ---------------------------

def real_time_prediction(model, actions, sequence_length, threshold=0.8):
    """Run real-time sign prediction using the trained model and display only the recognized word."""
    sequence = []
    last_action = ""
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Extract keypoints and update sequence
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]

            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                action_detected = actions[np.argmax(res)]
                
                # If the probability exceeds threshold, update last_action
                if res[np.argmax(res)] > threshold:
                    last_action = action_detected

            # Display only the recognized word
            cv2.rectangle(image, (0, 0), (640, 40), (0, 0, 0), -1)
            cv2.putText(image, last_action, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Real-Time Sign Prediction', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------
# 7. Main Routine with Argument Parsing
# ---------------------------

if __name__ == '_main_':
    parser = argparse.ArgumentParser(description="Indian Sign Language Recognition Pipeline")
    parser.add_argument("--mode", type=str, choices=["collect", "train", "predict"], required=True,
                        help="Mode: collect data, train model, or run real-time prediction")
    parser.add_argument("--actions", type=str, nargs='+', default=['hello', 'thanks', 'iloveyou'],
                        help="List of actions (signs) to detect")
    parser.add_argument("--no_sequences", type=int, default=30, help="Number of video sequences per action")
    parser.add_argument("--sequence_length", type=int, default=30, help="Number of frames per sequence")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--data_path", type=str, default="MP_Data", help="Path to save/load data")
    parser.add_argument("--model_path", type=str, default="action.keras", help="Path to save/load model")
    args = parser.parse_args()
    
    # Create folders for data collection if mode is 'collect'
    if args.mode == "collect":
        create_data_folders(args.data_path, np.array(args.actions), args.no_sequences)
        collect_data(args.data_path, np.array(args.actions), args.no_sequences, args.sequence_length)
    
    # Training mode: load data and train the model
    elif args.mode == "train":
        all_actions = os.listdir(args.data_path)
        X, y, label_map = load_data(args.data_path, np.array(args.all_actions), args.no_sequences, args.sequence_length)
        print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
        model, X_test, y_test = train_model(X, y, args.epochs, model_path=args.model_path)
        # Optionally, evaluate the model
        yhat = model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat_labels = np.argmax(yhat, axis=1).tolist()
        print("Confusion Matrix:")
        print(multilabel_confusion_matrix(ytrue, yhat_labels))
        print("Accuracy Score:", accuracy_score(ytrue, yhat_labels))
    
    # Real-time prediction mode: load the saved model and run predictions
    elif args.mode == "predict":
        if not os.path.exists(args.model_path):
            print("Trained model not found. Please run in train mode first.")
        else:
            model = load_model(args.model_path)
            real_time_prediction(model, np.array(args.actions), args.sequence_length)
=======
        return image


st.title("Real-Time Sign Language Detection")
webrtc_streamer(key="sign-lang", video_transformer_factory=SignLanguageTransformer)
>>>>>>> 8fcbc16 (yes)
