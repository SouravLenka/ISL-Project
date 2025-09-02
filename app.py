import os
import subprocess
import streamlit as st
import joblib
import cv2
import numpy as np

# Paths
MODEL_PATH = "model.pkl"

# ----------------------------
# Helper: Ensure model exists
# ----------------------------
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ No trained model found. Training the model now...")

        # Run training script
        result = subprocess.run(
            ["python", "train_classifier.py"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            st.error("❌ Training failed. Please check train_classifier.py")
            st.text(result.stderr)
            st.stop()
        else:
            st.success("✅ Model trained successfully!")

# ----------------------------
# Load Model
# ----------------------------
ensure_model()
model = joblib.load(MODEL_PATH)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="ISL Gesture Recognition", page_icon="🖐️", layout="centered")

st.title("🖐️ ISL Gesture Recognition (Inference)")

st.write("This app recognizes Indian Sign Language gestures using a trained classifier.")

# ----------------------------
# Upload Image / Video
# ----------------------------
uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Uploaded Image")

    # Preprocess as needed
    # (Change this according to your feature extraction pipeline)
    features = image.flatten().reshape(1, -1)

    # Prediction
    prediction = model.predict(features)[0]
    st.success(f"✅ Predicted Gesture: **{prediction}**")
