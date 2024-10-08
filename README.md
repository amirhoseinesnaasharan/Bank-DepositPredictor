# Bank-DepositPredictor

# SVM Classification with Bank Dataset

This project demonstrates the application of Support Vector Machines (SVM) on a bank dataset for binary classification. The dataset contains customer information and the objective is to predict whether a client will subscribe to a term deposit (`yes` or `no`).

## Features
- **Data Preprocessing**: 
  - Converts categorical target labels (`yes`, `no`) into binary values (1, 0).
  - Separates the dataset into numerical and categorical features.
  - Normalizes numerical features using `StandardScaler`.
  - Encodes categorical features using `OneHotEncoder`.
- **Feature Importance**:
  - Calculates the Pearson correlation matrix.
  - Identifies and removes the least important feature.
  - Identifies and plots the two most important features.
- **Model Training and Evaluation**:
  - Trains two SVM models: 
    - A linear kernel SVM.
    - A radial basis function (RBF) kernel SVM.
  - Evaluates models using accuracy, F1-score, precision, recall, and confusion matrix.