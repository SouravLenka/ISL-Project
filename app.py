import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import os

MODEL_FILE = "./model.p"

st.title("✋ ISL Gesture Recognition (Inference)")

# Check if trained model exists
if not os.path.exists(MODEL_FILE):
    st.error("❌ No trained model found. Please run train_classifier.py first.")
    st.stop()

# Load model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

st.write("Show a hand gesture to the camera below:")

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
        st.success(f"✅ Predicted gesture: **Class {prediction}**")

        # Show captured image
        st.image(frame, channels="BGR", caption=f"Predicted: Class {prediction}")

    else:
        st.warning("⚠️ No hand detected. Try again.")
