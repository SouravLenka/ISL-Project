import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.set_page_config(page_title="ISL Detection", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>🤟 Indian Sign Language Detection</h1>", 
    unsafe_allow_html=True
)
st.write("This app uses MediaPipe + OpenCV to detect hand gestures for ISL.")

# Camera input
img_file_buffer = st.camera_input("📷 Capture an image")

if img_file_buffer is not None:
    # Read image
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    # Convert to BGR for OpenCV
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Init Mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Process frame
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        st.success("✋ Hand detected!")
    else:
        st.warning("No hand detected. Try again!")

    # Show output
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels="RGB", caption="Processed Frame")
