import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import os

MODEL_FILE = "./model.p"

# Mapping from numeric label → alphabet
# ⚠️ Adjust this mapping if your dataset has fewer/more classes
label_map = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
}

st.title("✋ ISL Gesture → Alphabet Recognition")

# Check if trained model exists
if not os.path.exists(MODEL_FILE):
    st.error("❌ No trained model found. Please run train_classifier.py first.")
    st.stop()

# Load model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

st.write("📷 Show a hand gesture to the camera below:")

# Take input from webcam
img_file = st.camera_input("Capture gesture")

if img_file is not None:
    # Convert to OpenCV format
    image = Image.open(img_file)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preprocess with mediapipe
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        # Predict gesture
        prediction = model.predict([landmarks])[0]
        predicted_letter = label_map.get(prediction, f"Unknown (Class {prediction})")

        st.success(f"✅ Predicted gesture: **{predicted_letter}**")
    else:
        st.warning("⚠️ No hand detected. Try again.")
