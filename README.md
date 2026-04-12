# 🛒 E-commerce Purchase Prediction System

## 📌 Project Overview
This project predicts whether a user will make a purchase based on session and behavioral data using Machine Learning. The goal is to help businesses identify potential buyers and improve conversion strategies.

---

## 🚀 Key Features
- End-to-end Machine Learning Pipeline using sklearn Pipeline  
- Data preprocessing using ColumnTransformer  
- Hyperparameter tuning using GridSearchCV  
- Handling class imbalance  
- Threshold tuning to improve model performance  
- Interactive web app built using Streamlit  
- Deployed model for real-time predictions  

---

## 🧠 Machine Learning Workflow
1. Data preprocessing (encoding + scaling)  
2. Train-test split with stratification  
3. Model training using Decision Tree  
4. Hyperparameter tuning using GridSearchCV  
5. Model evaluation using Precision, Recall, F1-score  
6. Performance improvement:
   - Threshold tuning  
   - Class balancing  
7. Final model selection based on F1-score  

---

## 📊 Model Performance Comparison

| Model Version | Precision | Recall | F1 Score | Accuracy |
|--------------|----------|--------|----------|----------|
| Base Model | 0.77 | 0.46 | 0.58 | 0.90 |
| Threshold Tuned (0.4) ⭐ | 0.68 | 0.67 | **0.67** | 0.90 |
| Class Balanced Model | 0.50 | 0.83 | 0.62 | 0.84 |

---

## ✅ Final Model Selection
The base model showed high precision but low recall, missing many actual buyers.  
Threshold tuning improved the balance between precision and recall, achieving the highest F1-score.  
Therefore, the threshold-tuned model was selected as the final model.

---

## 🌐 Deployment
The model is deployed using Streamlit for real-time predictions.

👉 Live App: (Add your Streamlit link here after deployment)

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  

---

## 📂 Project Structure
├── app.py
├── model.pkl
├── threshold.pkl
├── columns.pkl
├── cat_values.pkl
├── requirements.txt
├── shop_smart.ipynb
├── README.md


---

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

💡 Key Learnings
Importance of handling class imbalance
Trade-off between precision and recall
Practical use of threshold tuning
Building end-to-end ML pipelines
Deploying ML models using Streamlit
📌 Future Improvements
Try advanced models (Random Forest, XGBoost)
Improve UI/UX of the application
Add feature importance visualization
