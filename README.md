E-commerce Purchase Prediction System
📌 Project Overview

This project predicts whether a user will make a purchase based on session-level behavioral data using Machine Learning. The objective is to help businesses identify high-intent users, improve targeting strategies, and increase conversion rates.

🚀 Key Features
Structured ML workflow from preprocessing to deployment
Feature preprocessing using scaling and encoding (saved for inference)
Hyperparameter tuning using GridSearchCV
Handling class imbalance using class weights
Decision threshold tuning using predicted probabilities
Model comparison across multiple configurations
Streamlit-based web app for real-time predictions
🧠 Machine Learning Workflow
Data preprocessing (encoding + scaling)
Train-test split with stratification
Model training using Decision Tree
Hyperparameter tuning using GridSearchCV
Model evaluation using Precision, Recall, F1-score
🔧 Performance Improvements
Threshold tuning to balance precision and recall
Class imbalance handling using class weights
Comparison of multiple model variants

Preprocessing components (scaler and encoder) are saved and reused during deployment to ensure consistency between training and inference.

📊 Model Performance Comparison
Model Version	Precision	Recall	F1 Score	Accuracy
Base Model	0.77	0.46	0.58	0.90
Threshold Tuned (0.4) ⭐	0.68	0.67	0.67	0.90
Class Balanced Model	0.50	0.83	0.62	0.84
✅ Final Model Selection
The base model showed high precision but low recall, missing many actual buyers
Threshold tuning improved the balance between precision and recall
The threshold-tuned model achieved the highest F1-score
Therefore, the threshold-tuned model was selected as the final model
🌐 Deployment

The model is deployed using Streamlit, where trained model and preprocessing artifacts are loaded to perform real-time predictions based on user input.

👉 Live App: (Add your link here)

🛠️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
Streamlit
Joblib
📂 Project Structure
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
▶️ How to Run Locally
pip install -r requirements.txt
streamlit run app.py
💡 Key Learnings
Importance of handling class imbalance in real-world datasets
Trade-off between precision and recall in classification problems
Practical implementation of threshold tuning using predicted probabilities
Saving and reusing preprocessing artifacts for deployment
Building and deploying ML models using Streamlit
📌 Future Improvements
Experiment with advanced models (Random Forest, Gradient Boosting, XGBoost)
Improve UI/UX of the Streamlit application
Add feature importance visualization
Integrate real-time data pipeline for production use
