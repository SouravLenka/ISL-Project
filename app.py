import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

st.set_page_config(page_title="ISL Project", layout="wide")

st.title("🤟 Indian Sign Language Detection")
st.write("This app uses MediaPipe + OpenCV to detect hand gestures for ISL.")


# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def process_frame(frame, hands_model):
    """Process a frame and return annotated frame"""
    frame = cv2.flip(frame, 1)  # mirror effect
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands_model.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    return frame


def main():
    run = st.checkbox("Start Camera")

    if run:
        stframe = st.empty()

        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to grab frame")
                    break

                frame = process_frame(frame, hands)

                # Convert BGR → RGB for Streamlit
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
    else:
        st.info("☝️ Check 'Start Camera' to begin.")


if __name__ == "__main__":
    main()
