#Here is the model that achieved the best results, after multiple attempts to find a way to improve the initial model the one which improved it was boostrapping

import pandas as pd
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
train_mse_scores = []
test_mse_scores = []
train_r2_scores = []
test_r2_scores = []

# Bootstrapping: Perform 5 iterations
n_iterations = 3
for _ in range(n_iterations):
    # Sample with replacement from the original dataset
    X_resampled, y_resampled = resample(X, y, random_state=42)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train the model on the resampled training data
    best_model.fit(X_train, y_train)

    # Predict on the training set
    y_train_pred = best_model.predict(X_train)
    # Predict on the testing set
    y_test_pred = best_model.predict(X_test)

    # Calculate MSE and R² scores for the training set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Calculate MSE and R² scores for the testing set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Append the scores to the lists
    train_mse_scores.append(train_mse)
    test_mse_scores.append(test_mse)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)

# Calculate the average MSE and R² scores for the training set
average_train_mse = sum(train_mse_scores) / len(train_mse_scores)
average_train_r2 = sum(train_r2_scores) / len(train_r2_scores)

# Calculate the average MSE and R² scores for the testing set
average_test_mse = sum(test_mse_scores) / len(test_mse_scores)
average_test_r2 = sum(test_r2_scores) / len(test_r2_scores)

print("Bootstrapping Results for Random Forest:")
print(f'Average Train MSE Score: {average_train_mse:.5f}')
print(f'Average Train R² Score: {average_train_r2:.5f}')
print(f'Average Test MSE Score: {average_test_mse:.5f}')
print(f'Average Test R² Score: {average_test_r2:.5f}')


#Location 2 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('Location2.csv')

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
best_params = {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt',
               'min_samples_leaf': 2, 'min_samples_split': 7, 'n_estimators': 398}

# Initialize the Random Forest model with the best parameters
best_model = RandomForestRegressor(**best_params, random_state=42)

# Initialize lists to store the scores
train_mse_scores = []
test_mse_scores = []
train_r2_scores = []
test_r2_scores = []

# Bootstrapping: Perform 5 iterations
n_iterations = 3
for _ in range(n_iterations):
    # Sample with replacement from the original dataset
    X_resampled, y_resampled = resample(X, y, random_state=42)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train the model on the resampled training data
    best_model.fit(X_train, y_train)

    # Predict on the training set
    y_train_pred = best_model.predict(X_train)
    # Predict on the testing set
    y_test_pred = best_model.predict(X_test)

    # Calculate MSE and R² scores for the training set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Calculate MSE and R² scores for the testing set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Append the scores to the lists
    train_mse_scores.append(train_mse)
    test_mse_scores.append(test_mse)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)

# Calculate the average MSE and R² scores for the training set
average_train_mse = sum(train_mse_scores) / len(train_mse_scores)
average_train_r2 = sum(train_r2_scores) / len(train_r2_scores)

# Calculate the average MSE and R² scores for the testing set
average_test_mse = sum(test_mse_scores) / len(test_mse_scores)
average_test_r2 = sum(test_r2_scores) / len(test_r2_scores)

print("Bootstrapping Results for Random Forest:")
print(f'Average Train MSE Score: {average_train_mse:.5f}')
print(f'Average Train R² Score: {average_train_r2:.5f}')
print(f'Average Test MSE Score: {average_test_mse:.5f}')
print(f'Average Test R² Score: {average_test_r2:.5f}')

# Location 3 

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('Location3.csv')

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
best_params = {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt',
               'min_samples_leaf': 2, 'min_samples_split': 7, 'n_estimators': 398}

# Initialize the Random Forest model with the best parameters
best_model = RandomForestRegressor(**best_params, random_state=42)

# Initialize lists to store the scores
train_mse_scores = []
test_mse_scores = []
train_r2_scores = []
test_r2_scores = []

# Bootstrapping: Perform 5 iterations
n_iterations = 3
for _ in range(n_iterations):
    # Sample with replacement from the original dataset
    X_resampled, y_resampled = resample(X, y, random_state=42)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train the model on the resampled training data
    best_model.fit(X_train, y_train)

    # Predict on the training set
    y_train_pred = best_model.predict(X_train)
    # Predict on the testing set
    y_test_pred = best_model.predict(X_test)

    # Calculate MSE and R² scores for the training set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Calculate MSE and R² scores for the testing set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Append the scores to the lists
    train_mse_scores.append(train_mse)
    test_mse_scores.append(test_mse)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)

# Calculate the average MSE and R² scores for the training set
average_train_mse = sum(train_mse_scores) / len(train_mse_scores)
average_train_r2 = sum(train_r2_scores) / len(train_r2_scores)

# Calculate the average MSE and R² scores for the testing set
average_test_mse = sum(test_mse_scores) / len(test_mse_scores)
average_test_r2 = sum(test_r2_scores) / len(test_r2_scores)

print("Bootstrapping Results for Random Forest:")
print(f'Average Train MSE Score: {average_train_mse:.5f}')
print(f'Average Train R² Score: {average_train_r2:.5f}')
print(f'Average Test MSE Score: {average_test_mse:.5f}')
print(f'Average Test R² Score: {average_test_r2:.5f}')

# Location 4

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('Location4.csv')

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
best_params = {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt',
               'min_samples_leaf': 2, 'min_samples_split': 7, 'n_estimators': 398}

# Initialize the Random Forest model with the best parameters
best_model = RandomForestRegressor(**best_params, random_state=42)

# Initialize lists to store the scores
train_mse_scores = []
test_mse_scores = []
train_r2_scores = []
test_r2_scores = []

# Bootstrapping: Perform 5 iterations
n_iterations = 3
for _ in range(n_iterations):
    # Sample with replacement from the original dataset
    X_resampled, y_resampled = resample(X, y, random_state=42)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train the model on the resampled training data
    best_model.fit(X_train, y_train)

    # Predict on the training set
    y_train_pred = best_model.predict(X_train)
    # Predict on the testing set
    y_test_pred = best_model.predict(X_test)

    # Calculate MSE and R² scores for the training set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Calculate MSE and R² scores for the testing set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Append the scores to the lists
    train_mse_scores.append(train_mse)
    test_mse_scores.append(test_mse)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)

# Calculate the average MSE and R² scores for the training set
average_train_mse = sum(train_mse_scores) / len(train_mse_scores)
average_train_r2 = sum(train_r2_scores) / len(train_r2_scores)

# Calculate the average MSE and R² scores for the testing set
average_test_mse = sum(test_mse_scores) / len(test_mse_scores)
average_test_r2 = sum(test_r2_scores) / len(test_r2_scores)

print("Bootstrapping Results for Random Forest:")
print(f'Average Train MSE Score: {average_train_mse:.5f}')
print(f'Average Train R² Score: {average_train_r2:.5f}')
print(f'Average Test MSE Score: {average_test_mse:.5f}')
print(f'Average Test R² Score: {average_test_r2:.5f}')

