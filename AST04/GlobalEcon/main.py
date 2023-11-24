# Dataset: https://www.kaggle.com/datasets/joebeachcapital/medical-insurance-costs

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# data_set = pd.read_csv('insurance.csv')
data_set = pd.read_csv('test_data.csv')

# Append a column of 1's first 
data_set.insert(0, 'x0', 1)

# Continuous real-valued Input Features: age, bmi, # of children
# X = data_set.drop('charges', axis=1)  
X = data_set.drop('LifeExpBirth', axis=1)  

# Output characteristics: Insurance costs (dollars)
# Y = data_set['charges']  
Y = data_set['LifeExpBirth']  

# 80% of the dataset for training and 20% for testing your model.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert to numpy for matrix operations
X_train_np = X_train.to_numpy()
Y_train_np = Y_train.to_numpy()
X_test_np = X_test.to_numpy()
Y_test_np = Y_test.to_numpy()

# w = (X_train^T * X_train)^(-1) * X_train^T * y_train
X_train_T = np.matrix.transpose(X_train_np)
X_train_T_X_train = np.matmul(X_train_T, X_train_np)
X_train_T_X_train_inverse = np.linalg.inv(X_train_T_X_train)
OLS_params = np.matmul(np.matmul(X_train_T_X_train_inverse, X_train_T), Y_train_np)

print(OLS_params)

# Get the predicted outputs 
Y_train_pred = np.matmul(X_train_np, OLS_params)
Y_test_pred = np.matmul(X_test_np, OLS_params)

# Evaluate the model
mse_train = mean_squared_error(Y_train_np, Y_train_pred)
mse_test = mean_squared_error(Y_test_np, Y_test_pred)

mae_train = mean_absolute_error(Y_train_np, Y_train_pred)
mae_test = mean_absolute_error(Y_test_np, Y_test_pred)

r2_train = r2_score(Y_train_np, Y_train_pred)
r2_test = r2_score(Y_test_np, Y_test_pred)

print("\nOLS Train MSE:", mse_train)
print("OLS Test MSE:", mse_test)

print("\nOLS Train MAE:", mae_train)
print("OLS Test MAE:", mae_test)

print("\nOLS Train R² Score:", r2_train)
print("OLS Test R² Score:", r2_test)

# Load the dataset
# data_set = pd.read_csv('insurance.csv')
data_set = pd.read_csv('test_data.csv')

# Continuous real-valued Input Features: age, bmi, # of children
# X = data_set.drop('charges', axis=1)  
X = data_set.drop('LifeExpBirth', axis=1)  

# Output characteristics: Insurance costs (dollars)
# Y = data_set['charges']  
Y = data_set['LifeExpBirth']   

# 80% of the dataset for training and 20% for testing your model.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the model using SGDRegressor and default parameters
pipeline = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
pipeline.fit(X_train, Y_train)

# Predicting with the trained model
Y_train_pred_sgd = pipeline.predict(X_train)
Y_test_pred_sgd = pipeline.predict(X_test)

# Calculate the evaluation metrics for SGD
mse_train_sgd = mean_squared_error(Y_train, Y_train_pred_sgd)
mse_test_sgd = mean_squared_error(Y_test, Y_test_pred_sgd)

mae_train_sgd = mean_absolute_error(Y_train, Y_train_pred_sgd)
mae_test_sgd = mean_absolute_error(Y_test, Y_test_pred_sgd)

r2_train_sgd = r2_score(Y_train, Y_train_pred_sgd)
r2_test_sgd = r2_score(Y_test, Y_test_pred_sgd)

# Print the results
print("\nSGD Train MSE:", mse_train_sgd)
print("SGD Test MSE:", mse_test_sgd)
print("\nSGD Train MAE:", mae_train_sgd)
print("SGD Test MAE:", mae_test_sgd)
print("\nSGD Train R² Score:", r2_train_sgd)
print("SGD Test R² Score:", r2_test_sgd)

# Convert dictionary to DataFrame
df = pd.DataFrame(data_set)

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
