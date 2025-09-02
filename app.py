# app.py
import os
import io
import sys
import json
import time
import pickle
import joblib
import subprocess
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import mediapipe as mp


# -----------------------------
# Config & paths
# -----------------------------
st.set_page_config(page_title="ISL Gesture Recognition", page_icon="🤟", layout="centered")

DATASET_PATH = "data.pickle"                   # created by create_dataset.py
MODEL_CANDIDATES = ["model.p", "model.pkl", "model.pickle"]
LABELS_PATH_CANDIDATES = ["labels.p", "labels.pkl", "labels.pickle"]  # optional (if you saved a LabelEncoder)
TRAIN_SCRIPT = "train_classifier.py"           # your existing training script


# -----------------------------
# Small utils
# -----------------------------
def find_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def load_model() -> Optional[object]:
    """Try to load a model saved via joblib or pickle from common filenames."""
    model_path = find_first_existing(MODEL_CANDIDATES)
    if not model_path:
        return None

    # Try joblib first, then pickle
    try:
        return joblib.load(model_path)
    except Exception:
        pass
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model '{model_path}': {e}")
        return None


def load_label_encoder():
    """Optional: load a LabelEncoder if you saved one separately."""
    lp = find_first_existing(LABELS_PATH_CANDIDATES)
    if not lp:
        return None
    try:
        with open(lp, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def ensure_trained_model():
    """
    If no trained model is present but dataset exists, offer a button to run your training script.
    (Note: Streamlit Cloud storage is ephemeral; trained file persists only for this session unless committed to repo.)
    """
    model = load_model()
    if model is not None:
        return model

    if not os.path.exists(DATASET_PATH):
        st.error("No trained model found, and 'data.pickle' is missing. "
                 "Please run data collection & create_dataset.py locally, train, and commit the model file.")
        st.stop()

    st.warning("No trained model found. You can train it now from the existing dataset.")
    if st.button("🛠️ Train model now"):
        with st.spinner("Training… this may take a minute."):
            # Run your training script; show stderr/stdout if it fails
            result = subprocess.run([sys.executable, TRAIN_SCRIPT], capture_output=True, text=True)
            if result.returncode != 0:
                st.error("Training failed. See error output below.")
                st.code(result.stderr or result.stdout)
                st.stop()
            else:
                st.success("Training finished. Loading model…")
                # tiny wait for filesystem flush
                time.sleep(0.5)

    # Load after potential training
    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Make sure your training script saves one of: "
                 f"{', '.join(MODEL_CANDIDATES)}")
        st.stop()
    return model


def pil_image_to_bgr(img: Image.Image) -> np.ndarray:
    """Handle EXIF orientation and convert to OpenCV BGR array."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img)           # RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def extract_features_from_landmarks(hand_landmarks) -> List[float]:
    """
    Build feature vector from MediaPipe hand landmarks.
    Common approach: use x,y for 21 points (ignore z), normalized by subtracting min x and y.
    Returns a list of length 42.
    """
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    min_x, min_y = min(xs), min(ys)
    features: List[float] = []
    for x, y in zip(xs, ys):
        features.extend([x - min_x, y - min_y])
    return features


def predict_gesture(model, features: List[float], label_encoder=None):
    """Return predicted label and (if available) probability/confidence."""
    # scikit-learn models usually accept 2D arrays
    X = np.asarray(features, dtype=np.float32).reshape(1, -1)

    try:
        y_pred = model.predict(X)
        label = y_pred[0]
    except Exception as e:
        st.error(f"Model predict failed: {e}")
        st.stop()

    # If a label encoder is present and model predicts indices, inverse_transform
    if label_encoder is not None:
        try:
            label = label_encoder.inverse_transform([label])[0]
        except Exception:
            pass

    # Try to get confidence if classifier supports predict_proba or decision_function
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            df = model.decision_function(X)
            if df.ndim == 1:
                # binary case
                df = np.vstack([-df, df]).T
            confidence = float(np.max(_softmax(df)[0]))
        except Exception:
            pass

    return label, confidence


def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


# -----------------------------
# UI
# -----------------------------
st.title("🤟 ISL Gesture Recognition")
st.caption("MediaPipe → Hand landmarks → scikit-learn classifier")

with st.expander("ℹ️ How to use", expanded=False):
    st.markdown(
        "- Click **Take photo** below to capture a frame from your webcam.\n"
        "- The app will detect your hand, extract landmarks, and predict the gesture.\n"
        "- If no model is found, you can train it (requires `data.pickle`).\n"
        "- On Streamlit Cloud, files written at runtime are **ephemeral**."
    )

# Load / ensure a trained model
model = ensure_trained_model()
label_encoder = load_label_encoder()

# Camera input (works in browser, also on Streamlit Cloud)
img_file = st.camera_input("📷 Take a photo", help="Use a clear background and good lighting.")

if img_file is not None:
    # Read and convert image to OpenCV BGR
    img = Image.open(img_file)
    frame_bgr = pil_image_to_bgr(img)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # MediaPipe hands (single-image mode for camera snapshots)
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6
    ) as hands:

        results = hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            st.warning("No hand detected. Try again with your hand centered and well lit.")
            st.image(frame_rgb, caption="Captured", use_container_width=True)
            st.stop()

        # Assume first hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks for display
        annotated = frame_bgr.copy()
        mp_draw.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Features → prediction
        features = extract_features_from_landmarks(hand_landmarks)
        label, confidence = predict_gesture(model, features, label_encoder)

        # Overlay text
        overlay = annotated.copy()
        text = f"{label}" if confidence is None else f"{label}  ({confidence*100:.1f}%)"
        cv2.rectangle(overlay, (10, 10), (10 + 320, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
        cv2.putText(annotated, text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

        st.success(f"**Prediction:** {text}")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected & classified", use_container_width=True)

else:
    st.info("👆 Use the **Take photo** button above to capture a frame.")
