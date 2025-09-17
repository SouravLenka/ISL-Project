''''
import pickle
import numpy as np

LABEL_MAP = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E"
    # Extend if you have more classes
}

def load_model(model_path="./model.p"):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def extract_landmarks(hand_landmarks):
    """Extract (x, y, z) coordinates from a Mediapipe hand landmarks object."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def predict(model, landmarks):
    """Predict the class using trained model and return readable label."""
    X = np.asarray(landmarks, dtype=np.float32).reshape(1, -1)
    pred = model.predict(X)[0]
    return LABEL_MAP.get(pred, f"Unknown (Class {pred})")
'''
import numpy as np
import pickle

# Fixed label mapping for your gestures
LABEL_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E"
    # Extend if you have more classes
}

def extract_landmarks(hand_landmarks):
    """
    Convert Mediapipe hand landmarks to (x, y, z) tuples.
    
    Args:
        hand_landmarks: Mediapipe hand landmarks object.
    
    Returns:
        List of (x, y, z) tuples for each landmark.
    """
    if not hand_landmarks or not hasattr(hand_landmarks, "landmark"):
        raise ValueError("Invalid hand_landmarks input.")
    
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]


def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks relative to wrist (first landmark).
    Handles flattened arrays, list of tuples, 2D or 3D coordinates.
    
    Args:
        landmarks: List of (x, y) or (x, y, z) tuples, or flattened array.
    
    Returns:
        Flattened numpy array of normalized landmarks.
    """
    landmarks = np.array(landmarks)

    if landmarks.size == 0:
        raise ValueError("Empty landmarks array.")

    # If 1D flattened array
    if landmarks.ndim == 1:
        if len(landmarks) % 3 == 0:
            landmarks = landmarks.reshape(-1, 3)
        elif len(landmarks) % 2 == 0:
            landmarks = landmarks.reshape(-1, 2)
        else:
            raise ValueError(f"Landmarks length {len(landmarks)} is not divisible by 2 or 3.")

    # Validate number of coordinates per point
    coords = landmarks.shape[1]
    if coords not in [2, 3]:
        raise ValueError(f"Expected 2 or 3 columns for landmarks, got {coords}.")

    # Normalize relative to wrist
    wrist = landmarks[0]
    normalized = landmarks - wrist

    return normalized.flatten()


def load_model(model_path="./model_rf.p"):
    """
    Load RandomForest model. LABEL_MAP is fixed.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model  # LABEL_MAP is fixed globally


def predict(model, landmarks):
    """
    Predict gesture and return correct alphabet using fixed LABEL_MAP.
    
    Args:
        model: Trained scikit-learn model.
        landmarks: normalized landmarks
    
    Returns:
        Predicted alphabet
    """
    if landmarks is None or len(landmarks) == 0:
        raise ValueError("Invalid landmarks for prediction.")
    
    pred_int = model.predict([landmarks])[0]
    return LABEL_MAP.get(pred_int, "?")  # Returns '?' if prediction not in LABEL_MAP
