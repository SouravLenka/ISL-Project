import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import streamlit as st
import os

DATA_PICKLE = "./data.pickle"
MODEL_FILE = "./model_rf.p"  # Updated to match app.py

st.title("🤖 Train Random Forest Classifier")

# Check if model already exists
if os.path.exists(MODEL_FILE):
    st.success("✅ model_rf.p already exists. Skipping training.")
    st.stop()

# Check if dataset exists
if not os.path.exists(DATA_PICKLE):
    st.error("❌ No dataset found. Please run create_dataset.py first.")
    st.stop()

# Load dataset
data_dict = pickle.load(open(DATA_PICKLE, "rb"))
data = np.asarray(data_dict["data"])       # Expecting list of landmark arrays
labels = np.asarray(data_dict["labels"])

st.info(f"📊 Loaded {len(labels)} samples.")
st.write("Class distribution:", dict(Counter(labels)))

# Normalize landmarks relative to wrist (first point in each sample)
def normalize_landmarks(sample):
    wrist_x, wrist_y = sample[0][0], sample[0][1]
    normalized = [(x - wrist_x, y - wrist_y) for x, y in sample]
    return np.array(normalized).flatten()

data_normalized = np.array([normalize_landmarks(s) for s in data])

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data_normalized, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

st.info("⏳ Training RandomForestClassifier...")
progress = st.progress(0)

# Initialize and train Random Forest
model = RandomForestClassifier(
    n_estimators=150,      # Can tune higher for better performance
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(x_train, y_train)
progress.progress(70)

# Evaluate
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
progress.progress(100)

# Save the trained model
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

st.success(f"✅ Training completed with accuracy: {acc:.2f}")
st.write("Class distribution in predictions:", dict(Counter(y_pred)))
