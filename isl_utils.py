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
    # Use x, y, z for compatibility with your dataset
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def predict(model, landmarks):
    X = np.asarray(landmarks, dtype=np.float32).reshape(1, -1)
    pred = model.predict(X)[0]
    return LABEL_MAP.get(pred, f"Unknown (Class {pred})")