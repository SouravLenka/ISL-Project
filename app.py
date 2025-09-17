'''
import streamlit as st
import os
import sys
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageOps

# ✅ Updated Mediapipe import for 0.10.x
from mediapipe import solutions as mp_solutions

from isl_utils import load_model, extract_landmarks, predict

st.set_page_config(page_title="ISL Gesture Recognition", page_icon="🤟", layout="centered")

DATASET_PATH = "data.pickle"
MODEL_FILE = "model.p"
TRAIN_SCRIPT = "train_classifier.py"

st.title("🤟 ISL Gesture Recognition")
st.caption("MediaPipe → Hand landmarks → scikit-learn classifier")

with st.expander("ℹ️ How to use", expanded=False):
    st.markdown(
        "- Click **Take photo** below to capture a frame from your webcam.\n"
        "- The app will detect your hand, extract landmarks, and predict the gesture.\n"
        "- If no model is found, you can train it (requires `data.pickle`).\n"
        "- On Streamlit Cloud, files written at runtime are **ephemeral**."
    )


def ensure_trained_model():
    if os.path.exists(MODEL_FILE):
        return load_model(MODEL_FILE)
    if not os.path.exists(DATASET_PATH):
        st.error("❌ No dataset found. Please run data collection and dataset creation first.")
        st.stop()
    st.warning("No trained model found. You can train it now from the existing dataset.")
    if st.button("🛠️ Train model now"):
        with st.spinner("Training model..."):
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
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


model = ensure_trained_model()

img_file = st.camera_input("📷 Take a photo", help="Use a clear background and good lighting.")

if img_file is not None:
    img = Image.open(img_file)
    frame_bgr = pil_image_to_bgr(img)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ✅ Updated mediapipe usage
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
        predicted_letter = predict(model, landmarks)
        st.success(f"Prediction: **{predicted_letter}**")
else:
    st.info("👆 Use the **Take photo** button above to capture a frame.")
'''
import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from mediapipe import solutions as mp_solutions
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from isl_utils import extract_landmarks, normalize_landmarks, predict  # ✅ Import utilities

DATASET_PATH = "data.pickle"
MODEL_FILE = "model_rf.p"

st.set_page_config(page_title="ISL Gesture Recognition", page_icon="🤟", layout="centered")

st.title("🤟 ISL Gesture Recognition")
st.caption("MediaPipe → Hand landmarks → Random Forest classifier")

with st.expander("ℹ️ How to use", expanded=False):
    st.markdown(
        "- Click **Take photo** below to capture a frame from your webcam.\n"
        "- The app will detect your hand, extract landmarks, normalize them, and predict the gesture.\n"
        "- If no model is found, the app will automatically train it from `data.pickle`.\n"
        "- On Streamlit Cloud, files written at runtime are **ephemeral**."
    )

def pil_image_to_bgr(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def train_model():
    """Train Random Forest classifier from dataset"""
    if not os.path.exists(DATASET_PATH):
        st.error("❌ No dataset found. Please create dataset first.")
        st.stop()

    st.info("⏳ Training Random Forest model automatically...")
    data_dict = pickle.load(open(DATASET_PATH, "rb"))
    data = np.asarray(data_dict["data"])
    labels = np.asarray(data_dict["labels"])

    st.write("Class distribution:", dict(Counter(labels)))

    # Normalize all landmarks using utility function
    data_normalized = np.array([normalize_landmarks(s) for s in data])

    x_train, x_test, y_train, y_test = train_test_split(
        data_normalized, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )

    progress = st.progress(0)
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    progress.progress(70)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    progress.progress(100)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    st.success(f"✅ Training completed with accuracy: {acc:.2f}")
    return model

def ensure_trained_model():
    """Load the model if exists; otherwise train it"""
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return train_model()

# Load or train model automatically
model = ensure_trained_model()

# Camera input
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
