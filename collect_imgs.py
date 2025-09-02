import os
import streamlit as st
from PIL import Image
import numpy as np
import cv2

DATA_DIR = "./data"
number_of_classes = 5
dataset_size = 100

st.title("📸 Collect Images for Dataset")

# Skip if dataset already exists
if os.path.exists(DATA_DIR) and any(os.listdir(DATA_DIR)):
    st.success("✅ Data already exists. Skipping collection.")
    st.stop()

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Loop through classes
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    st.header(f"Collecting data for class {j}")
    st.info(f"Take {dataset_size} pictures for class {j}. Press the camera button below.")

    count = 0
    while count < dataset_size:
        img_file = st.camera_input(f"Class {j}: Take picture {count+1}/{dataset_size}")

        if img_file is not None:
            image = Image.open(img_file)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            save_path = os.path.join(class_dir, f"{count}.jpg")
            cv2.imwrite(save_path, frame)

            count += 1
            st.success(f"Saved image {count}/{dataset_size} for class {j}")

    st.success(f"✅ Completed data collection for class {j}")

st.balloons()
st.success("🎉 Data collection completed for all classes!")
