# Dataset: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_set = pd.read_csv('diabetes_prediction_dataset.csv')

# Continuous real-valued Input Features: age, bmi, HbA1c_level, blood_glucose_level
X = data_set.drop('diabetes', axis=1)  

# Output characteristics: Diabetes (positive/negative)
Y = data_set['diabetes']  

# 80% of the dataset for training and 20% for testing your model.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Train the logistic regression model using SGDClassifier and loss set to 'log_loss'
model = make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42))
model.fit(X_train, Y_train)

# Get model steps
sgd_classifier = model.named_steps['sgdclassifier']

# Coefficients
coefficients = sgd_classifier.coef_[0]

# Bias
bias = sgd_classifier.intercept_[0]

# Print Equation
feature_names = X.columns
equation = "f(x) = " + str(bias)
for feature_name, coef in zip(feature_names, coefficients):
    equation += f" + ({coef})*{feature_name}"
print(f"\n{equation}")


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

# Convert dictionary to DataFrame
df = pd.DataFrame(data_set)

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()



