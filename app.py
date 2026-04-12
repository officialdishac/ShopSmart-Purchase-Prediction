import streamlit as st
import pandas as pd
import pickle
from joblib import load

# Load model (joblib)
model = load("model.joblib")

# Load other files
with open("threshold.pkl", "rb") as f:
    threshold = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

with open("cat_values.pkl", "rb") as f:
    cat_values = pickle.load(f)

# Title
st.title("🛒 E-commerce Purchase Prediction")

st.write("Enter customer details below:")

# Create input fields
input_data = {}

# Extract numeric columns from pipeline
num_cols = model.named_steps["preprocess"].transformers_[0][2]

for col in columns:
    
    if col in num_cols:
        input_data[col] = st.number_input(f"{col}", value=0.0)
    
    elif col in cat_values:
        input_data[col] = st.selectbox(f"{col}", cat_values[col])
    
    else:
        # fallback (e.g., boolean columns like Weekend)
        input_data[col] = st.selectbox(f"{col}", [0, 1])

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict"):
    prob = model.predict_proba(input_df)[:, 1][0]
    prediction = 1 if prob > threshold else 0

    st.write(f"### Prediction Probability: {prob:.2f}")

    if prediction == 1:
        st.success("✅ Customer is likely to purchase")
    else:
        st.error("❌ Customer is NOT likely to purchase")