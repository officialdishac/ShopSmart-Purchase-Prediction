# ShopSmart Purchase Prediction (Decision Tree)

## Project Overview
This project builds a machine learning classification model to predict whether a visitor will complete a purchase on an e-commerce website based on session behavior data.

The dataset contains 12,330 user sessions with numerical and categorical features.

## Problem Statement
The objective is to predict the target variable **Revenue**:
- 1 → Purchase
- 0 → No Purchase

Since the dataset is imbalanced, **F1-score** is used as the primary evaluation metric.

## Machine Learning Approach

### Data Preprocessing
- Numerical features scaled using StandardScaler
- Categorical features encoded using OneHotEncoder
- Implemented using ColumnTransformer and Pipeline

### Model Used
- Decision Tree Classifier
  - max_depth controlled
  - min_samples_leaf tuned
  - class_weight="balanced"

### Hyperparameter Tuning
- GridSearchCV used for optimal parameter selection

## Model Evaluation
- F1 Score
- Classification Report
- Confusion Matrix

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (if used)

## How to Run the Project

1. Clone the repository
2. Place the dataset file inside the project folder
3. Install required libraries:
(or manually install pandas, numpy, scikit-learn, matplotlib)

4. Run the notebook or Python script
