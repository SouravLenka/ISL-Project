import os
import pickle
import cv2
import mediapipe as mp
import streamlit as st
from collections import Counter

DATA_DIR = "./data"
PICKLE_FILE = "./data.pickle"

st.title("🛠️ Create Dataset")

# Skip if data.pickle already exists
if os.path.exists(PICKLE_FILE):
    st.success("✅ data.pickle already exists. Skipping dataset creation.")
    st.stop()

if not os.path.exists(DATA_DIR) or not any(os.listdir(DATA_DIR)):
    st.error("❌ No data found. Please run collect_imgs.py first.")
    st.stop()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

dirs = os.listdir(DATA_DIR)
progress = st.progress(0)
total_dirs = len(dirs)
current_dir = 0

for dir_ in dirs:
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            data.append(landmarks)
            labels.append(int(dir_))

    current_dir += 1
    progress.progress(int((current_dir / total_dirs) * 100))

data_dict = {"data": data, "labels": labels}
with open(PICKLE_FILE, "wb") as f:
    pickle.dump(data_dict, f)

st.success(f"✅ Dataset created successfully with {len(labels)} samples.")
st.info(f"Class distribution: {Counter(labels)}")
