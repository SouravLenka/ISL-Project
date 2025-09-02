import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import streamlit as st
import os

DATA_PICKLE = "./data.pickle"
MODEL_FILE = "./model.p"

st.title("🤖 Train Classifier")

# Skip if model already exists
if os.path.exists(MODEL_FILE):
    st.success("✅ model.p already exists. Skipping training.")
    st.stop()

# Check if dataset is available
if not os.path.exists(DATA_PICKLE):
    st.error("❌ No dataset found. Please run create_dataset.py first.")
    st.stop()

# Load dataset
data_dict = pickle.load(open(DATA_PICKLE, "rb"))
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

st.info(f"📊 Loaded {len(labels)} samples.")
st.write("Class distribution:", dict(Counter(labels)))

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

st.info("⏳ Training RandomForestClassifier...")
progress = st.progress(0)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)
progress.progress(70)

# Evaluate
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
progress.progress(100)

# Save model
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

st.success(f"✅ Training completed with accuracy: {acc:.2f}")
