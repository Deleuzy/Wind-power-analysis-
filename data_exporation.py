# A short data exploration of the mean values in the first location 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.utils import resample

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

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Adding a constant term for statsmodels
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Train the model using statsmodels for detailed statistics
ols_model = sm.OLS(y_train, X_train_sm).fit()

# Get predictions and prediction intervals
predictions = ols_model.get_prediction(X_test_sm)
prediction_summary = predictions.summary_frame(alpha=0.05)  # 95% prediction interval

print("Statsmodels Prediction Summary:")
print(prediction_summary[['mean', 'mean_ci_lower', 'mean_ci_upper', 'obs_ci_lower', 'obs_ci_upper']])

# Bootstrap sampling to estimate prediction uncertainty
n_iterations = 1000
predictions = np.zeros((n_iterations, X_test.shape[0]))

for i in range(n_iterations):
    X_resampled, y_resampled = resample(X_train, y_train)
    model.fit(X_resampled, y_resampled)
    predictions[i, :] = model.predict(X_test)

# Calculate mean and confidence intervals of predictions
prediction_mean = predictions.mean(axis=0)
prediction_std = predictions.std(axis=0)
confidence_interval = 1.96 * prediction_std  # 95% confidence interval

print("\nBootstrap Prediction Summary:")
print(f'Mean predictions: {prediction_mean}')
print(f'Confidence intervals: {confidence_interval}')
