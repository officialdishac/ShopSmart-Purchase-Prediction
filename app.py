import streamlit as st
import pandas as pd
import pickle
from joblib import load

# Load model and preprocessor separately
model = load("model.joblib")
preprocessor = load("preprocessor.joblib")

with open("threshold.pkl", "rb") as f:
    threshold = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

with open("cat_values.pkl", "rb") as f:
    cat_values = pickle.load(f)

st.title("🛒 E-commerce Purchase Prediction")

st.write("Enter customer details below:")

input_data = {}

for col in columns:
    
    if col in cat_values:
        input_data[col] = st.selectbox(f"{col}", cat_values[col])
    else:
        input_data[col] = st.number_input(f"{col}", value=0.0)

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    # Apply preprocessing manually
    processed = preprocessor.transform(input_df)

    prob = model.predict_proba(processed)[:, 1][0]
    prediction = 1 if prob > threshold else 0

    st.write(f"### Prediction Probability: {prob:.2f}")

    if prediction == 1:
        st.success("✅ Customer is likely to purchase")
    else:
        st.error("❌ Customer is NOT likely to purchase")