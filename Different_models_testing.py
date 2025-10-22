# Here we test differnet models and see which one has the best chance of improvement - in this case was Random Forest 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf'),
    'XGBoost': XGBRegressor(),
    'KNN': KNeighborsRegressor()
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    mse_train = mean_squared_error(y_train, train_predictions)
    r2_train = r2_score(y_train, train_predictions)
    mse_test = mean_squared_error(y_test, test_predictions)
    r2_test = r2_score(y_test, test_predictions)
    results[name] = {'Train MSE': mse_train, 'Train R^2': r2_train, 'Test MSE': mse_test, 'Test R^2': r2_test}

# Print results
print("Model Performance Comparison:")
for name, result in results.items():
    print(f"--- {name} ---")
    print(f"Train MSE: {result['Train MSE']:.2f}, Train R^2: {result['Train R^2']:.2f}")
    print(f"Test MSE: {result['Test MSE']:.2f}, Test R^2: {result['Test R^2']:.2f}")
    print()
