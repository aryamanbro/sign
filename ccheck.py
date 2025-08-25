# ccheck.py

import os
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    """
    Processes an image with the MediaPipe Holistic model.

    Args:
        image: The input image (BGR format).
        model: The MediaPipe Holistic model instance.

    Returns:
        A tuple containing the processed image (BGR) and the detection results.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color space
    image.flags.writeable = False  # Make image non-writeable for performance
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Make image writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image, results


def draw_styled_landmarks(image, results):
    """
    Draws styled landmarks and connections on the image, checking for their existence first.

    Args:
        image: The image to draw on.
        results: The MediaPipe detection results.
    """
    # Draw face connections if they exist
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
    # Draw pose connections if they exist
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    # Draw left hand connections if they exist
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    # Draw right hand connections if they exist
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )


def extract_keypoints(results):
    """
    Extracts keypoints from the detection results into a flattened numpy array.

    Args:
        results: The MediaPipe detection results.

    Returns:
        A numpy array containing the concatenated keypoints. Returns an array of zeros if no landmarks are detected.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


def create_data_folders(DATA_PATH, actions, no_sequences):
    """
    Creates the directory structure for storing collected data.
    Structure: DATA_PATH / action_name / sequence_number

    Args:
        DATA_PATH (str): The root path for the data.
        actions (list): A list of action names (strings).
        no_sequences (int): The number of sequences to collect for each action.
    """
    for action in actions:
        for sequence in range(no_sequences):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)


def load_and_prepare_data(DATA_PATH, actions, no_sequences, sequence_length):
    """
    Loads the collected keypoint data from disk and prepares it for training.

    Args:
        DATA_PATH (str): The root path where the data is stored.
        actions (list): A list of action names.
        no_sequences (int): The number of sequences collected for each action.
        sequence_length (int): The number of frames in each sequence.

    Returns:
        A tuple (X, y) where X is the feature data and y is the one-hot encoded labels.
    """
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"), allow_pickle=False)
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return X, y


def build_and_train_model(X_train, y_train, X_test, y_test, epochs, actions):
    """
    Builds, compiles, and trains the LSTM model.

    Args:
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        epochs (int): The number of epochs to train for.
        actions (list): List of action names to determine the output layer size.

    Returns:
        The trained Keras model.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    return model
