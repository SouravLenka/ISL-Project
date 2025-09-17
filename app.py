import streamlit as st
import os
import sys
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageOps
from mediapipe import solutions as mp_solutions

from isl_utils import load_model, extract_landmarks, predict

st.set_page_config(page_title="ISL Gesture Recognition", page_icon="🤟", layout="centered")

DATASET_PATH = "data.pickle"
MODEL_FILE = "model_rf.p"  # Updated model file name for Random Forest
TRAIN_SCRIPT = "train_classifier.py"

st.title("🤟 ISL Gesture Recognition")
st.caption("MediaPipe → Hand landmarks → Random Forest classifier")

with st.expander("ℹ️ How to use", expanded=False):
    st.markdown(
        "- Click **Take photo** below to capture a frame from your webcam.\n"
        "- The app will detect your hand, extract landmarks, normalize them, and predict the gesture.\n"
        "- If no model is found, you can train it (requires `data.pickle`).\n"
        "- On Streamlit Cloud, files written at runtime are **ephemeral**."
    )

def ensure_trained_model():
    """
    Load a trained model if exists. If not, allow the user to train it using TRAIN_SCRIPT.
    Returns:
        model (RandomForestClassifier): Trained Random Forest model
    """
    if os.path.exists(MODEL_FILE):
        return load_model(MODEL_FILE)

    if not os.path.exists(DATASET_PATH):
        st.error("❌ No dataset found. Please run data collection and dataset creation first.")
        st.stop()

    st.warning("No trained model found. You can train it now from the existing dataset.")
    if st.button("🛠️ Train model now"):
        with st.spinner("Training Random Forest model..."):
            result = subprocess.run([sys.executable, TRAIN_SCRIPT], capture_output=True, text=True)
            st.text(result.stdout)
            if result.returncode != 0:
                st.error(f"Training failed:\n{result.stderr}")
                st.stop()

    if os.path.exists(MODEL_FILE):
        return load_model(MODEL_FILE)

    st.error("Model training failed or was not completed.")
    st.stop()


def pil_image_to_bgr(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to OpenCV BGR format
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks relative to the wrist (first landmark)
    """
    wrist_x, wrist_y = landmarks[0][0], landmarks[0][1]
    normalized = [(x - wrist_x, y - wrist_y) for x, y in landmarks]
    return np.array(normalized).flatten()


model = ensure_trained_model()

img_file = st.camera_input("📷 Take a photo", help="Use a clear background and good lighting.")

if img_file is not None:
    img = Image.open(img_file)
    frame_bgr = pil_image_to_bgr(img)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Mediapipe hand detection
    with mp_solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6
    ) as hands_detector:
        results = hands_detector.process(frame_rgb)
        if not results.multi_hand_landmarks:
            st.warning("No hand detected. Try again.")
            st.stop()

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = extract_landmarks(hand_landmarks)
        normalized_landmarks = normalize_landmarks(landmarks)
        predicted_letter = predict(model, normalized_landmarks)
        st.success(f"Prediction: **{predicted_letter}**")
else:
    st.info("👆 Use the **Take photo** button above to capture a frame.")
