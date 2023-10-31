# Dataset: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss

# Load the dataset
data_set = pd.read_csv('diabetes_prediction_dataset.csv')

# Continuous real-valued Input Features: age, bmi, HbA1c_level, blood_glucose_level
X = data_set.drop('diabetes', axis=1)  

# Output characteristics: Diabetes (positive/negative)
Y = data_set['diabetes']  

# 80% of the dataset for training and 20% for testing your model.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the model, fit the model using training data 
model = GaussianNB()
model.fit(X_train, Y_train)

# Predictions
Y_train_pred = model.predict(X_train)
Y_train_prob = model.predict_proba(X_train)[:, 1]  # Probabilities for the positive class
Y_test_pred = model.predict(X_test)
Y_test_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Evaluate the model
print("\nPrecise Training Accuracy:", accuracy_score(Y_train, Y_train_pred))
print("\nPrecise Testing Accuracy:", accuracy_score(Y_test, Y_test_pred))
print("\nClassification Report (Test Data):\n", classification_report(Y_test, Y_test_pred))
print("\nClassification Report (Training Data):\n", classification_report(Y_train, Y_train_pred))
print("\nLog Loss (Test Data):", log_loss(Y_test, Y_test_prob))
print("\nLog Loss (Training Data):", log_loss(Y_train, Y_train_prob))

