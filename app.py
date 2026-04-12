import streamlit as st
import pandas as pd
import pickle
from joblib import load
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load model
model = load("model.joblib")

# Load metadata
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

st.title("🛒 E-commerce Purchase Prediction")

input_data = {}

for col in columns:
    if col in cat_cols:
        input_data[col] = st.selectbox(col, cat_values[col])
    else:
        input_data[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

if st.button("Predict"):

    # 🔥 Recreate preprocessing manually
    num_df = input_df[num_cols]
    cat_df = input_df[cat_cols]

    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    num_scaled = scaler.fit_transform(num_df)
    cat_encoded = encoder.fit_transform(cat_df)

    import numpy as np
    processed = np.hstack([num_scaled, cat_encoded])

    prob = model.predict_proba(processed)[:, 1][0]
    prediction = 1 if prob > threshold else 0

    st.write(f"### Prediction Probability: {prob:.2f}")

    if prediction == 1:
        st.success("✅ Customer is likely to purchase")
    else:
        st.error("❌ Customer is NOT likely to purchase")