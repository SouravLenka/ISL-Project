import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile

st.set_page_config(page_title="ISL Detection", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>🤟 Indian Sign Language Detection</h1>", 
    unsafe_allow_html=True
)
st.write("This app uses MediaPipe + OpenCV to detect hand gestures for ISL.")

# Checkbox for camera
start_cam = st.checkbox("📷 Start Camera")

# Placeholder for video
frame_placeholder = st.empty()

if start_cam:
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to access camera.")
            break

        # Flip for selfie view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
