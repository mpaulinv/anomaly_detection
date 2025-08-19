import numpy as np 
import pandas as pd
import os
import logging
import io 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import mlflow
import mlflow.sklearn
import mlflow.statsmodels
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor


from config import FEATURED_TRAIN_PATH, WINDOW_SIZE, THRESHOLD_MULTIPLIER, LOG_MODEL_SELECTION_PATH
from utils import load_data, train_model_naive, test_model_naive, plot_validation_with_anomalies, plot_validation_vs_predictions, log_comprehensive_model_metrics, comprehensive_anomaly_analysis

os.makedirs("results", exist_ok=True)

# Initialize MLflow
mlflow.set_experiment("Anomaly_Detection_Comparison")

# Load the train data after feature engineering
train_features = load_data(FEATURED_TRAIN_PATH, log_path=LOG_MODEL_SELECTION_PATH)

# Optional: Inspect the loaded features
print(train_features.head())
train_features.info()
train_features.describe()

# First we split the data into training and validation sets. For the validation set I will use around the last 20% of the data, that is the last 5 days.
# Find the latest timestamp
train_features['timestamp'] = pd.to_datetime(train_features['timestamp'])
latest = train_features['timestamp'].max()

# Select rows from the last 5 days
train = train_features[train_features['timestamp'] < (latest - pd.Timedelta(days=4))]
validation = train_features[train_features['timestamp'] >= (latest - pd.Timedelta(days=4))]

print(train.columns)
print(validation.columns)

# We begin with the first model that was proposed based on the 60 min rolling window and three standard deviations around the mean. 
# I will start with the same implementation from the code which uses the last rolling window in the test set. 


#normal_mean, normal_std = train_model_naive(train, WINDOW_SIZE)
#anomalies = test_model_naive(validation, normal_mean, normal_std, THRESHOLD_MULTIPLIER)

# Plot validation data and anomalies
#plot_validation_with_anomalies(validation, anomalies, value_col='value', timestamp_col='timestamp', save_path="results" + '/validation_anomalies_naive.png')

# The naive model performs rather poorly since it is using the fixed window. Hence it detects everything outside of the hour of the window as anomalies. 
# I want to try a simple regression model first to see if I can improve on the naive approach. 


# Assume train_features is already loaded and has sin_hour, cos_hour, and lag features
#feature_cols = ['sin_hour', 'cos_hour', 'value_lag_1', 'value_lag_2', 'value_lag_3', 'hour', 'minute', 'dayofweek']
#feature_cols = ['sin_hour', 'cos_hour', 'minute']
feature_cols = ['sin_hour', 'cos_hour', 'minute']
X_train = train[feature_cols].dropna()
y_train = train.loc[X_train.index, 'value']

# Fit linear model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")


# We want to check the distribution of the residuals (mean which should be zero, std and APE)

# Calculate residuals and APE
y_pred_train = model.predict(X_train)
residuals_train = y_train - y_pred_train
std_resid_train = residuals_train.std()
ape_train = np.abs(residuals_train / y_pred_train) * 100

# Print statistics
print(f"Mean of residuals (train): {residuals_train.mean():.4f}")
print(f"Standard deviation of residuals (train): {std_resid_train:.4f}")
print(f"APE (train) - mean: {ape_train.mean():.2f}%, median: {np.median(ape_train):.2f}%, max: {ape_train.max():.2f}%")

# Optionally, plot the distribution of APE
plt.figure(figsize=(8,4))
plt.hist(ape_train, bins=50, alpha=0.7)
plt.xlabel('Absolute Percentage Error (%)')
plt.ylabel('Frequency')
plt.title('Distribution of APE in Training Data')
plt.tight_layout()
plt.show()


# Predict on validation set
X_val = validation[feature_cols].dropna()
y_val = validation.loc[X_val.index, 'value']
y_pred = model.predict(X_val)

# Log Linear Regression model to MLflow
linear_params = {
    'features': feature_cols,
    'model_type': 'LinearRegression',
    'n_features': len(feature_cols)
}

# Calculate anomalies for logging
residuals_val = y_val - y_pred
ape_val = np.abs(residuals_val / y_pred) * 100
std_resid_train = residuals_train.std()
ape_threshold = 2.0
anomaly_mask_std = np.abs(residuals_val) > (2.5 * std_resid_train)
anomalies_std = validation.loc[X_val.index][anomaly_mask_std]

# Log to MLflow using comprehensive function
log_comprehensive_model_metrics("Linear_Regression", y_val, y_pred, anomalies_std, validation, 
                               value_col='value', timestamp_col='timestamp',
                               model_params=linear_params, model_object=model)

plot_validation_vs_predictions(validation, y_pred, value_col='value', timestamp_col='timestamp', save_path=None)


# --- Dual Threshold Anomaly Detection ---
# Assume y_val and y_pred are defined for validation set
residuals_val = y_val - y_pred
ape_val = np.abs(residuals_val / y_pred) * 100

# Thresholds from training
std_resid_train = 0.2264  # Replace with your computed value
ape_threshold = 2.0  # 2%

# 3 std residual threshold
anomaly_mask_std = np.abs(residuals_val) > (2.5 * std_resid_train)
anomalies_std = validation.loc[X_val.index][anomaly_mask_std]
print(f"Anomalies (2.5 std): {anomalies_std.shape[0]}")

# 2% APE threshold
anomaly_mask_ape = ape_val > ape_threshold
anomalies_ape = validation.loc[X_val.index][anomaly_mask_ape]
print(f"Anomalies (APE > 2%): {anomalies_ape.shape[0]}")

plot_validation_with_anomalies(validation, anomalies_std, value_col='value', timestamp_col='timestamp', save_path="results" + '/validation_anomalies_linear.png')


print(train.columns)
print(validation.columns)



# isolation forest 
# Use the same lag features for train and validation
iso_feature_cols = ['sin_hour', 'cos_hour', 'minute', 'value_lag_1', 'value_lag_2', 'value_lag_3']
X_train_iso = train[iso_feature_cols].dropna()
X_val_iso = validation[iso_feature_cols].dropna()

# Fit Isolation Forest on training data
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train_iso)

# Predict anomalies on validation set
iso_pred = iso_forest.predict(X_val_iso)  # -1 for anomaly, 1 for normal
anomalies_iso = validation.loc[X_val_iso.index][iso_pred == -1]
print(f"Anomalies (Isolation Forest): {anomalies_iso.shape[0]}")

# Log Isolation Forest to MLflow using comprehensive function
iso_params = {
    'contamination': 0.01,
    'features': iso_feature_cols,
    'model_type': 'IsolationForest',
    'random_state': 42
}

# Log using comprehensive function (no predictions for pure anomaly detection)
log_comprehensive_model_metrics("Isolation_Forest", None, None, anomalies_iso, validation,
                               value_col='value', timestamp_col='timestamp',
                               model_params=iso_params, model_object=iso_forest)




#  ADAPTIVE ROLLING WINDOW APPROACH
# Use different window sizes based on time of day/week patterns
def adaptive_anomaly_detection(data, timestamp_col='timestamp', value_col='value'):
    """Adaptive rolling window that learns different patterns for different times"""
    data = data.copy()
    data['hour'] = data[timestamp_col].dt.hour
    data['dayofweek'] = data[timestamp_col].dt.dayofweek
    
    # Group by hour and day to learn hour-specific patterns
    anomalies = []
    
    for hour in range(24):
        hour_data = data[data['hour'] == hour].copy()
        if len(hour_data) < 10:  # Skip if not enough data
            continue
            
        # Use rolling quantiles instead of mean/std
        hour_data['rolling_median'] = hour_data[value_col].rolling(window=10, center=True).median()
        hour_data['rolling_q25'] = hour_data[value_col].rolling(window=10, center=True).quantile(0.25)
        hour_data['rolling_q75'] = hour_data[value_col].rolling(window=10, center=True).quantile(0.75)
        hour_data['iqr'] = hour_data['rolling_q75'] - hour_data['rolling_q25']
        
        # Adaptive threshold based on IQR
        hour_data['lower_bound'] = hour_data['rolling_q25'] - 2 * hour_data['iqr']
        hour_data['upper_bound'] = hour_data['rolling_q75'] + 2 * hour_data['iqr']
        
        # Detect anomalies
        hour_anomalies = hour_data[
            (hour_data[value_col] < hour_data['lower_bound']) | 
            (hour_data[value_col] > hour_data['upper_bound'])
        ]
        anomalies.append(hour_anomalies)
    
    return pd.concat(anomalies) if anomalies else pd.DataFrame()

# Apply adaptive approach
anomalies_adaptive = adaptive_anomaly_detection(validation)
print(f"Anomalies (Adaptive Rolling): {len(anomalies_adaptive)}")

# Log Adaptive Rolling to MLflow
adaptive_params = {
    'model_type': 'Adaptive_Rolling_Window',
    'window_size': 10,
    'threshold_method': 'IQR',
    'iqr_multiplier': 2
}

log_comprehensive_model_metrics("Adaptive_Rolling", None, None, anomalies_adaptive, validation,
                               value_col='value', timestamp_col='timestamp',
                               model_params=adaptive_params, model_object=None)

if len(anomalies_adaptive) > 0:
    plot_validation_with_anomalies(validation, anomalies_adaptive, 
                                  save_path="results/validation_anomalies_adaptive.png")

# 2. RANDOM FOREST APPROACH
# Learn temporal patterns without assuming functional form
rf_feature_cols = ['hour', 'minute']
available_rf_cols = [col for col in rf_feature_cols if col in train.columns]

if available_rf_cols:
    X_train_rf = train[available_rf_cols].dropna()
    y_train_rf = train.loc[X_train_rf.index, 'value']
    
    # Random Forest learns complex temporal patterns
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_rf, y_train_rf)
    
    # Predict on validation
    X_val_rf = validation[available_rf_cols].dropna()
    y_val_rf = validation.loc[X_val_rf.index, 'value']
    y_pred_rf = rf_model.predict(X_val_rf)
    
    # Anomaly detection based on residuals
    rf_residuals = y_val_rf - y_pred_rf
    rf_std = rf_residuals.std()
    
    rf_anomaly_mask = np.abs(rf_residuals) > (2.5 * rf_std)
    anomalies_rf = validation.loc[X_val_rf.index][rf_anomaly_mask]
    
    print(f"Anomalies (Random Forest): {len(anomalies_rf)}")
    
    # Log Random Forest to MLflow
    rf_params = {
        'n_estimators': 100,
        'features': available_rf_cols,
        'model_type': 'RandomForestRegressor',
        'random_state': 42,
        'anomaly_threshold': '3_std'
    }
    
    log_comprehensive_model_metrics("Random_Forest", y_val_rf, y_pred_rf, anomalies_rf, validation,
                                   value_col='value', timestamp_col='timestamp',
                                   model_params=rf_params, model_object=rf_model)
    
    if len(anomalies_rf) > 0:
        plot_validation_with_anomalies(validation, anomalies_rf, 
                                      save_path="results/validation_anomalies_rf.png")

# 3. TEMPORAL-AWARE ONLINE ANOMALY DETECTION
# Uses only past data for each prediction (respects temporal order)
def temporal_anomaly_detection(data, timestamp_col='timestamp', value_col='value', window=25):
    """
    Temporal-aware anomaly detection that only uses past data.
    For each point, compares it to recent historical statistics.
    """
    data = data.copy().sort_values(timestamp_col)
    anomalies = []
    
    for i in range(window, len(data)):  # Start after minimum window
        current_row = data.iloc[i]
        current_value = current_row[value_col]
        
        # Only use past data (temporal awareness)
        historical_window = data.iloc[max(0, i-window):i][value_col]
        
        if len(historical_window) >= 10:  # Minimum data for statistics
            hist_mean = historical_window.mean()
            hist_std = historical_window.std()
            
            # Check if current value is anomalous compared to recent history
            if hist_std > 0:
                z_score = abs(current_value - hist_mean) / hist_std
                if z_score > 3:  # 3-sigma rule
                    anomalies.append(current_row.to_frame().T)
    
    return pd.concat(anomalies) if anomalies else pd.DataFrame()

# Apply temporal-aware approach
anomalies_temporal = temporal_anomaly_detection(validation, window=25)
print(f"Anomalies (Temporal-Aware): {len(anomalies_temporal)}")

# Log Temporal-Aware to MLflow
temporal_params = {
    'model_type': 'Temporal_Aware_Online',
    'window_size': 25,
    'threshold_method': '3_sigma_historical',
    'min_history_points': 10
}

log_comprehensive_model_metrics("Temporal_Aware_Online", None, None, anomalies_temporal, validation,
                               value_col='value', timestamp_col='timestamp',
                               model_params=temporal_params, model_object=None)

if len(anomalies_temporal) > 0:
    plot_validation_with_anomalies(validation, anomalies_temporal, 
                                  save_path="results/validation_anomalies_temporal.png")



### Model selection comments. I tried a few approaches, each with advantages and disadvantages.
### Linear regression model: An interpretable model that relies only on the lags, but the relationship between the target and variables relies on linearity. 
### Isolation forest: Non-supervised method that relies on the easiness of separation of anomalies from the normal data points.
### Random forest: Uses the lags and the features to capture complex interactions and non-linearities in the data. 
### Temporal-aware online anomaly detection: Similar to the original approach but with a few corrections. 1) It uses only past data points to evaluate the anomaly (no future data leakage). 2) Window size was adjusted to 25 minutes in the past.

### To compare across models I use a few metrics based on the anomalies identified. 1) Number and share of anomalies detected. In this I tried to make the best models have a similar number of anomalies identified to be able to compare. 
#   2) Plot of anomalies vs data to visualy assess whether the model is capturing real outliers or just regular data points. 
#   3) Metrics: Comparing three metrics one is the local separation, that is the distance between the anomalies and their closest six neighbors. Temporal separation: Distance between the anomaly and neighbors within a 30 minute ratio. Temporal spread: This is not a metric to "optimize" against but rather 
#   it can help us identify if the model is only capturing anomalies in certain hours or if they are generally spread around. Based on exploratory analysis I would expect anomalies to be spread around. 
# 
#   The results are logged and available in mlflow. 
#   Based on the results I have selected the random forest model. 
#   The isolation forest does not perform well in this context, due to a low degree of separation between anomalies and normal data points even if anomalies identified are half of those from other models
#   The other three models have similar performance at around 62-81 anomalies identified or a rate of anomalies of 1-1.5%. For the final model I can reduce increase the sensitivity in production or reduce based on risk appetite and investigation capacity. 
#   Of these models I find the following advantages and disadvantages:
#   - Linear Regression: Easy to interpret, but depends strongly on the data following the specific sinoidal functional form. If we are looking for anomalies where the sensor readings deviate from this pattern it will perform well. 
#   - however, if there is no real physical or business reason why this should be the case, the model may require significant re-engineering to perform well. 
#   - Random Forest: Good performance and can capture non-linearities. It has few features which makes the model usage in production simple. It can be quickly retrained if the data changes patterns. May not catch deviations from the functional form as quickly as the linear model. 
#   - Temporal-Aware: Addresses temporal aspects well. However, it suffers from some of the same deficiencies as the initally proposed solution, that is, outliers may mask future outliers. Also, it is not as easy to use in production 
#   - as it requires windows of observations to be fed on batch. 
#   - For these reasons and the better performance of the random forest on the metrics in the validation set, I have selected the random forest as the proposed solution. 
