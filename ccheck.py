import os
import numpy as np
import mediapipe as mp
import cv2
from streamlit_webrtc import VideoTransformerBase
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def create_data_folders(DATA_PATH, actions, no_sequences):
    """
    Create directory structure for data collection:
      DATA_PATH/{action}/{sequence}/
    """
    for action in actions:
        for seq in range(no_sequences):
            os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)


class KeypointCollector(VideoTransformerBase):
    """
    VideoTransformerBase that extracts and saves MediaPipe keypoints from webcam frames.
    Usage:
      collector = KeypointCollector(DATA_PATH, actions, no_sequences, sequence_length)
      collector.start_collection(action_index)
    """
    def __init__(self, DATA_PATH, actions, no_sequences, sequence_length):
        self.DATA_PATH = DATA_PATH
        self.actions = actions
        self.no_sequences = no_sequences
        self.sequence_length = sequence_length

        self.action_index = 0
        self.sequence = 0
        self.frame_num = 0
        self.collecting = False

        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def start_collection(self, action_index: int):
        """
        Begin data collection for the specified action index.
        Resets sequence and frame counters.
        """
        self.action_index = action_index
        self.sequence = 0
        self.frame_num = 0
        self.collecting = True

    def extract_keypoints(self, results):
        """
        Extract pose, face, left and right hand keypoints as a flat numpy array.
        """
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])

    def transform(self, frame):
        """
        Called for each video frame. Draws landmarks and, if collection is active,
        saves keypoints to .npy files under DATA_PATH.
        """
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)

        # Draw landmarks
        if results:
            mp_drawing.draw_landmarks(img, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Save keypoints if collecting
        if self.collecting:
            keypoints = self.extract_keypoints(results)
            action = self.actions[self.action_index]
            seq = self.sequence
            fn = self.frame_num
            save_dir = os.path.join(self.DATA_PATH, action, str(seq))
            np.save(os.path.join(save_dir, f"{fn}.npy"), keypoints)

            # Overlay status
            cv2.putText(img, f"Recording {action} Seq {seq+1}/{self.no_sequences} Frame {fn+1}/{self.sequence_length}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.frame_num += 1
            if self.frame_num >= self.sequence_length:
                self.frame_num = 0
                self.sequence += 1
                if self.sequence >= self.no_sequences:
                    self.collecting = False

        return img


# Data loading & training functions unchanged below

def load_data(DATA_PATH, actions, no_sequences, sequence_length):
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}
    for action in actions:
        for seq in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                arr = np.load(os.path.join(DATA_PATH, action, str(seq), f"{frame_num}.npy"))
                window.append(arr)
            sequences.append(window)
            labels.append(label_map[action])
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return X, y, label_map


def train_model(X, y, epochs, log_dir="Logs", model_path="action.h5"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    tb = TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, epochs=epochs, callbacks=[tb])
    model.save(model_path)
    return model, X_test, y_test


def real_time_prediction(model, actions, sequence_length, threshold=0.8):
    from streamlit_webrtc import webrtc_streamer
    from tensorflow.keras.models import load_model as lm

    class Predictor(VideoTransformerBase):
        def __init__(self):
            self.seq = []
            self.last = ""
            self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5,
                                                 min_tracking_confidence=0.5)

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img, results = mediapipe_detection(img, self.holistic)
            draw_styled_landmarks(img, results)
            keypoints = extract_keypoints(results)
            self.seq.append(keypoints)
            self.seq = self.seq[-sequence_length:]
            if len(self.seq) == sequence_length:
                res = model.predict(np.expand_dims(self.seq, 0))[0]
                if res.max() > threshold:
                    self.last = actions[np.argmax(res)]
            cv2.rectangle(img, (0,0), (640,40), (0,0,0), -1)
            cv2.putText(img, self.last, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            return img

    webrtc_streamer(key="pred", video_transformer_factory=Predictor,
                    media_stream_constraints={"video": True, "audio": False}, async_transform=True)
