# Dataset: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_set = pd.read_csv('diabetes.csv')

# Continuous real-valued Input Features: age, height, weight, ap_hi, ap_lo
X = data_set.drop('diabetes', axis=1)  

# Output characteristics: Diabetes (positive/negative)
Y = data_set['diabetes']  

# 80% of the dataset for training and 20% for testing your model.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# EXPERIMENT01: 3 layer, learning rate 0.01, relu
model1 = MLPClassifier(
    hidden_layer_sizes=(100, 100, 100),     # 3 hidden layers 100 nodes each
    learning_rate_init=0.01,                # learning rate 0.01
    activation='relu',                      # relu
    max_iter=300, verbose=True,             # print to console
    n_iter_no_change=10, tol=1e-4           # 10 epochs no improvement
)

# EXPERIMENT02: 2 hidden layers, learning rate 0.001, tanh
model2 = MLPClassifier(
    hidden_layer_sizes=(100, 100),                  # 2 hidden layers 100 nodes each
    learning_rate_init=0.001,                       # learning rate normal
    activation='tanh',                              # tanh
    max_iter=300, verbose=True,                     # print to console
    n_iter_no_change=30, tol=1e-4                   # 10 epochs no improvement
)

# EXPERIMENT03: early stopping, batch Size 64, relu
model3 = MLPClassifier(
    early_stopping=True,                # Enable early stopping
    batch_size=64,                      # Smaller batch size
    activation='relu',                  # relu
    max_iter=300, verbose=True,         # print to console
    n_iter_no_change=30, tol=1e-4       # 30 epochs no improvement
)

# Train and evaluate each model as done previously

model1.fit(X_train, Y_train)
model2.fit(X_train, Y_train)
model3.fit(X_train, Y_train)

# Predict on the training data
Y_train_pred = model1.predict(X_train)
# Get probabilities for each class
Y_train_prob = model1.predict_proba(X_train)

# Predict on the test data
Y_test_pred = model1.predict(X_test)
# Get probabilities for each class
Y_test_prob = model1.predict_proba(X_test)

# Evaluate the model
print("Model 1: 3 layer, learning rate 0.01, relu")
print("\nPrecise Training Accuracy:", accuracy_score(Y_train, Y_train_pred))
print("\nPrecise Testing Accuracy:", accuracy_score(Y_test, Y_test_pred))
print("\nClassification Report (Test Data):\n", classification_report(Y_test, Y_test_pred))
print("\nClassification Report (Training Data):\n", classification_report(Y_train, Y_train_pred))
print("\nLog Loss (Test Data):", log_loss(Y_test, Y_test_prob))
print("\nLog Loss (Training Data):", log_loss(Y_train, Y_train_prob))

# Predict on the training data
Y_train_pred = model2.predict(X_train)
# Get probabilities for each class
Y_train_prob = model2.predict_proba(X_train)

# Predict on the test data
Y_test_pred = model2.predict(X_test)
# Get probabilities for each class
Y_test_prob = model2.predict_proba(X_test)

# Evaluate the model
print("Model 2: 2 hidden layers, learning rate 0.001, tanh")
print("\nPrecise Training Accuracy:", accuracy_score(Y_train, Y_train_pred))
print("\nPrecise Testing Accuracy:", accuracy_score(Y_test, Y_test_pred))
print("\nClassification Report (Test Data):\n", classification_report(Y_test, Y_test_pred))
print("\nClassification Report (Training Data):\n", classification_report(Y_train, Y_train_pred))
print("\nLog Loss (Test Data):", log_loss(Y_test, Y_test_prob))
print("\nLog Loss (Training Data):", log_loss(Y_train, Y_train_prob))

# Predict on the training data
Y_train_pred = model3.predict(X_train)
# Get probabilities for each class
Y_train_prob = model3.predict_proba(X_train)

# Predict on the test data
Y_test_pred = model3.predict(X_test)
# Get probabilities for each class
Y_test_prob = model3.predict_proba(X_test)

# Evaluate the model
print("Model 3: early stopping, batch Size 64, relu")
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

