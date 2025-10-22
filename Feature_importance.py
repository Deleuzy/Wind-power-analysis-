# Here we see the feature importance of each of the feature against wind power to see which one is the most important
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Train the Random Forest regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importances for Random Forest
rf_feature_importances = rf_model.feature_importances_

# Plot feature importances for Random Forest
plt.figure(figsize=(10, 6))
plt.barh(X.columns, rf_feature_importances)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.show()

# Train the SVM regression model
svm_model = SVR(kernel='linear')  # Use linear kernel to extract feature importances
svm_model.fit(X_train, y_train)

# Feature importances for SVM (coefficients)
svm_feature_importances = np.abs(svm_model.coef_[0])

# Plot feature importances for SVM
plt.figure(figsize=(10, 6))
plt.barh(X.columns, svm_feature_importances)
plt.xlabel('Feature Importance (Absolute Coefficients)')
plt.title('SVM Feature Importances')
plt.show()

# Train the XGBoost regression model
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

# Feature importances for XGBoost
xgb_feature_importances = xgb_model.feature_importances_

# Plot feature importances for XGBoost
plt.figure(figsize=(10, 6))
plt.barh(X.columns, xgb_feature_importances)
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importances')
plt.show()

# Train the KNN regression model
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)

# KNN does not have feature importances inherently, so we skip plotting for KNN
