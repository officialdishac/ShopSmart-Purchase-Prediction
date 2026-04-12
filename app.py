import streamlit as st
import pandas as pd
import pickle
from joblib import load
import numpy as np

# ================================
# Load model
# ================================
model = load("model.joblib")

# ================================
# Load preprocessing objects
# ================================
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ================================
# Load metadata
# ================================
with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

with open("num_cols.pkl", "rb") as f:
    num_cols = pickle.load(f)

with open("cat_cols.pkl", "rb") as f:
    cat_cols = pickle.load(f)

with open("cat_values.pkl", "rb") as f:
    cat_values = pickle.load(f)

with open("threshold.pkl", "rb") as f:
    threshold = pickle.load(f)

# ================================
# UI
# ================================
st.title("🛒 E-commerce Purchase Prediction")
st.write("Enter customer details below:")

input_data = {}

for col in columns:
    if col in cat_cols:
        input_data[col] = st.selectbox(col, cat_values[col])
    else:
        input_data[col] = st.number_input(col, value=0.0)

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# ================================
# Prediction
# ================================
if st.button("Predict"):

    try:
        # Split features
        num_df = input_df[num_cols]
        cat_df = input_df[cat_cols]

        # ✅ Use TRAINED scaler & encoder (IMPORTANT FIX)
        num_scaled = scaler.transform(num_df)
        cat_encoded = encoder.transform(cat_df)

        # Combine features
        processed = np.hstack([num_scaled, cat_encoded])

        # Predict
        prob = model.predict_proba(processed)[:, 1][0]
        prediction = 1 if prob > threshold else 0

        # Output
        st.write(f"### Prediction Probability: {prob:.2f}")

        if prediction == 1:
            st.success("✅ Customer is likely to purchase")
        else:
            st.error("❌ Customer is NOT likely to purchase")

    except Exception as e:
        st.error(f"Error during prediction: {e}")