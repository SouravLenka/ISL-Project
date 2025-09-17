'''''
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

if os.path.exists(MODEL_FILE):
    st.success("✅ model.p already exists. Skipping training.")
    st.stop()

if not os.path.exists(DATA_PICKLE):
    st.error("❌ No dataset found. Please run create_dataset.py first.")
    st.stop()

data_dict = pickle.load(open(DATA_PICKLE, "rb"))
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

st.info(f"📊 Loaded {len(labels)} samples.")
st.write("Class distribution:", dict(Counter(labels)))

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

st.info("⏳ Training RandomForestClassifier...")
progress = st.progress(0)

model = RandomForestClassifier()
model.fit(x_train, y_train)
progress.progress(70)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
progress.progress(100)

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

st.success(f"✅ Training completed with accuracy: {acc:.2f}")

'''
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import streamlit as st
import os

from isl_utils import normalize_landmarks, LABEL_MAP

DATA_PICKLE = "./data.pickle"
MODEL_FILE = "./model_rf.p"

st.title("🤖 Train RandomForest Classifier")

if os.path.exists(MODEL_FILE):
    st.success("✅ model_rf.p already exists. Skipping training.")
    st.stop()

if not os.path.exists(DATA_PICKLE):
    st.error("❌ No dataset found. Please run create_dataset.py first.")
    st.stop()

# Load dataset
data_dict = pickle.load(open(DATA_PICKLE, "rb"))
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

# Map string labels to integers according to LABEL_MAP
label_to_int = {v: k for k, v in LABEL_MAP.items()}
labels_int = np.array([label_to_int[l] for l in labels])

st.info(f"📊 Loaded {len(labels)} samples.")
st.write("Class distribution:", dict(Counter(labels)))

# Normalize landmarks
st.info("⏳ Normalizing landmarks...")
data_normalized = np.array([normalize_landmarks(s) for s in data])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    data_normalized, labels_int, test_size=0.2, shuffle=True, stratify=labels_int
)

# Train RandomForest
st.info("⏳ Training RandomForestClassifier...")
progress = st.progress(0)
model = RandomForestClassifier()
model.fit(x_train, y_train)
progress.progress(70)

# Evaluate accuracy
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
progress.progress(100)

# Save model
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

st.success(f"✅ Training completed with accuracy: {acc:.2f}")
st.write("🎯 Model is now ready and will predict correct alphabets using LABEL_MAP.")
