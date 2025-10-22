# Looking at how well can the model explain the uncertainty of the results and we can see clear linear results both ways 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('Location1.csv')

# Ensure no missing values
data = data.dropna()

# Convert the 'Time' column to datetime
data['Time'] = pd.to_datetime(data['Time'])

# Extract the hour of the day as a feature
data['Hour'] = data['Time'].dt.hour

# Select features and target variable
X = data[['Hour', 'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 'windspeed_10m',
          'windspeed_100m', 'winddirection_10m', 'winddirection_100m', 'windgusts_10m']]
y = data['Power']

# Define the Random Forest parameters
best_params = {'bootstrap': False, 'max_depth': 25, 'max_features': 'sqrt',
               'min_samples_leaf': 2, 'min_samples_split': 7, 'n_estimators': 198}

# Initialize the Random Forest model with the best parameters
best_model = RandomForestRegressor(**best_params, random_state=42)

# Initialize lists to store the scores
test_mse_scores = []
test_r2_scores = []

# Bootstrapping: Perform 5 iterations
n_iterations = 3
bootstrap_preds = []

for _ in range(n_iterations):
    # Sample with replacement from the original dataset
    X_resampled, y_resampled = resample(X, y, random_state=42)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train the model on the resampled training data
    best_model.fit(X_train, y_train)

    # Predict on the testing set
    y_test_pred = best_model.predict(X_test)

    # Store predictions for each iteration
    bootstrap_preds.append(y_test_pred.reshape(-1, 1))

    # Calculate MSE and R² scores for the testing set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Append the scores to the lists
    test_mse_scores.append(test_mse)
    test_r2_scores.append(test_r2)

# Calculate the average MSE and R² scores for the testing set
average_test_mse = sum(test_mse_scores) / len(test_mse_scores)
average_test_r2 = sum(test_r2_scores) / len(test_r2_scores)

print("Bootstrapping Results for Random Forest:")
print(f'Average Test MSE Score: {average_test_mse:.5f}')
print(f'Average Test R² Score: {average_test_r2:.5f}')

# Concatenate prediction arrays
bootstrap_preds = np.concatenate(bootstrap_preds, axis=1)

# Calculate prediction intervals
lower = np.percentile(bootstrap_preds, 2.5, axis=1)
upper = np.percentile(bootstrap_preds, 97.5, axis=1)

# Plot prediction intervals
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', label='Actual vs. Predicted')
plt.errorbar(y_test, y_test_pred, yerr=[y_test_pred-lower, upper-y_test_pred], fmt='o', color='red', alpha=0.5, label='Prediction Interval (95%)')
plt.xlabel('Actual Power')
plt.ylabel('Predicted Power')
plt.title('Random Forest Prediction Intervals')
plt.legend()
plt.grid(True)
plt.show()

# The second shows the same but it draws lines between the predicted and actual 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('Location1.csv')

# Ensure no missing values
data = data.dropna()

# Convert the 'Time' column to datetime
data['Time'] = pd.to_datetime(data['Time'])

# Extract the hour of the day as a feature
data['Hour'] = data['Time'].dt.hour

# Select features and target variable
X = data[['Hour', 'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 'windspeed_10m',
          'windspeed_100m', 'winddirection_10m', 'winddirection_100m', 'windgusts_10m']]
y = data['Power']

# Define the Random Forest parameters
best_params = {'bootstrap': False, 'max_depth': 25, 'max_features': 'sqrt',
               'min_samples_leaf': 2, 'min_samples_split': 7, 'n_estimators': 198}

# Initialize the Random Forest model with the best parameters
best_model = RandomForestRegressor(**best_params, random_state=42)

# Initialize lists to store the scores
test_mse_scores = []
test_r2_scores = []

# Bootstrapping: Perform 5 iterations
n_iterations = 3
bootstrap_preds = []

for _ in range(n_iterations):
    # Sample with replacement from the original dataset
    X_resampled, y_resampled = resample(X, y, random_state=42)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train the model on the resampled training data
    best_model.fit(X_train, y_train)

    # Predict on the testing set
    y_test_pred = best_model.predict(X_test)

    # Store predictions for each iteration
    bootstrap_preds.append(y_test_pred.reshape(-1, 1))

    # Calculate MSE and R² scores for the testing set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Append the scores to the lists
    test_mse_scores.append(test_mse)
    test_r2_scores.append(test_r2)

# Calculate the average MSE and R² scores for the testing set
average_test_mse = sum(test_mse_scores) / len(test_mse_scores)
average_test_r2 = sum(test_r2_scores) / len(test_r2_scores)

print("Bootstrapping Results for Random Forest:")
print(f'Average Test MSE Score: {average_test_mse:.5f}')
print(f'Average Test R² Score: {average_test_r2:.5f}')

# Concatenate prediction arrays
bootstrap_preds = np.concatenate(bootstrap_preds, axis=1)

# Calculate prediction intervals
lower = np.percentile(bootstrap_preds, 2.5, axis=1)
upper = np.percentile(bootstrap_preds, 97.5, axis=1)

# Calculate the regression line
regression_line = np.mean(bootstrap_preds, axis=1)

# Calculate prediction interval differences
lower_diff = np.abs(regression_line - lower)
upper_diff = np.abs(upper - regression_line)

# Plot prediction intervals and regression line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', label='Actual vs. Predicted')
plt.errorbar(y_test, regression_line, yerr=[lower_diff, upper_diff], fmt='o', color='red', alpha=0.5, label='Prediction Interval (95%)')
plt.plot(y_test, regression_line, color='green', label='Regression Line')
plt.xlabel('Actual Power')
plt.ylabel('Predicted Power')
plt.title('Random Forest Regression with Prediction Intervals')
plt.legend()
plt.grid(True)
plt.show()
