# 🛒 E-commerce Purchase Prediction System

## 📌 Project Overview
This project predicts whether a user will make a purchase based on session-level behavioral data using Machine Learning. The goal is to help businesses identify high-intent users, optimize marketing strategies, and improve conversion rates.

The project follows a complete end-to-end ML pipeline, from data preprocessing and model training to deployment using a Streamlit web application.

---

## 🚀 Key Features
- End-to-end ML pipeline (preprocessing → training → evaluation → deployment)
- Hyperparameter tuning using GridSearchCV
- Handling class imbalance in real-world data
- Threshold optimization using predicted probabilities
- Model comparison across multiple approaches
- Reusable preprocessing pipeline for consistent inference
- Streamlit-based interactive web application for real-time predictions

---

## 🧠 Machine Learning Workflow

1. **Data Preprocessing**
   - Separation of numerical and categorical features
   - Scaling using StandardScaler
   - Encoding using OneHotEncoder

2. **Train-Test Split**
   - Stratified split to preserve class distribution

3. **Model Training**
   - Decision Tree Classifier used as baseline model

4. **Hyperparameter Tuning**
   - GridSearchCV used to optimize parameters such as:
     - max_depth  
     - min_samples_leaf  

5. **Model Evaluation**
   - Metrics used:
     - Precision  
     - Recall  
     - F1-score (primary metric due to class imbalance)

6. **Performance Improvement**
   - Class balancing to improve recall
   - Threshold tuning using predicted probabilities

7. **Final Model Strategy**
   - A hyperparameter-tuned Decision Tree is used as the final trained model
   - A custom decision threshold (0.4) is applied during inference to optimize F1-score

8. **Artifact Saving**
   - Model and preprocessing components are saved to ensure consistency during deployment

---

## 📊 Model Performance Comparison

| Model Version              | Precision | Recall | F1 Score | Accuracy |
|---------------------------|----------|--------|----------|----------|
| Base Model                | 0.77     | 0.46   | 0.58     | 0.90     |
| Threshold Applied (0.4) ⭐ | 0.68     | 0.67   | 0.67     | 0.90     |
| Class Balanced Model      | 0.50     | 0.83   | 0.62     | 0.84     |

---

## ✅ Final Model Selection

- The base model achieved high precision but low recall, missing many actual buyers  
- Class balancing improved recall but significantly reduced precision  
- Applying a custom probability threshold (0.4) provided the best balance between precision and recall  

**Final Approach:**
- Hyperparameter-tuned Decision Tree used as the trained model  
- Threshold tuning applied during inference using `predict_proba`  

---

## 🌐 Deployment

The model is deployed using **Streamlit** for real-time predictions.

### 🔄 Inference Pipeline
1. User inputs are collected via UI  
2. Inputs are converted into a structured DataFrame  
3. Preprocessing is applied using saved artifacts:
   - Scaler (for numerical features)
   - Encoder (for categorical features)  
4. Features are combined and passed to the model  
5. Model outputs probability using `predict_proba`  
6. Custom threshold (0.4) is applied to generate final prediction  

This ensures consistency between training and production environments.

👉 **Live App:** https://shopsmart-purchase-prediction.streamlit.app/

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  

---

## 📂 Project Structure

├── app.py  
├── model.joblib  
├── scaler.pkl  
├── encoder.pkl  
├── columns.pkl  
├── cat_values.pkl  
├── num_cols.pkl  
├── cat_cols.pkl  
├── threshold.pkl  
├── requirements.txt  
├── shop_smart.ipynb  
├── README.md  


---

## ▶️ How to Run Locally
pip install -r requirements.txt  
streamlit run app.py  

---

💡 Key Learnings
- Importance of handling class imbalance in real-world datasets
- Trade-off between precision and recall in classification problems
- Practical implementation of threshold tuning using predicted probabilities
- Saving and reusing preprocessing artifacts for deployment
- Building and deploying ML models using Streamlit

---
📌 Future Improvements
- Try advanced models (Random Forest, XGBoost)
- Improve UI/UX of the application
- Add feature importance visualization
