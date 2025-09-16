import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import os
from isl_utils import load_model, extract_landmarks, predict

MODEL_FILE = "./model.p"

st.title("✋ ISL Gesture → Alphabet Recognition")

if not os.path.exists(MODEL_FILE):
    st.error("❌ No trained model found. Please run train_classifier.py first.")
    st.stop()

model = load_model(MODEL_FILE)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

st.write("📷 Show a hand gesture to the camera below:")

img_file = st.camera_input("Capture gesture")

if img_file is not None:
    image = Image.open(img_file)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = extract_landmarks(hand_landmarks)
        predicted_letter = predict(model, landmarks)
        st.success(f"✅ Predicted gesture: **{predicted_letter}**")
    else:
        st.warning("⚠️ No hand detected. Try again.")