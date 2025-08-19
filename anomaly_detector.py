# anomaly_detector.py
#
# This script detects anomalies in sensor data. It calculates a rolling average
# and flags any point that is more than 3 standard deviations away.
# It's very basic and needs to be turned into a real product.

import pandas as pd
import numpy as np

# --- Hardcoded Configuration & Paths ---
# This should be configurable.
TRAIN_DATA_PATH = 'sensor_data_train.csv'
TEST_DATA_PATH = 'sensor_data_test.csv'
WINDOW_SIZE = 60  # Rolling window for stats
THRESHOLD_MULTIPLIER = 3.0 # Number of std deviations for a point to be an anomaly

print("--- Starting Anomaly Detection POC ---")

# --- Step 1: "Training" ---
# Load the training data to establish a baseline of normal behavior.
# This logic is tightly coupled and not reusable.
try:
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    train_df.set_index('timestamp', inplace=True)

    # Calculate rolling statistics on the training data
    rolling_mean = train_df['value'].rolling(window=WINDOW_SIZE).mean()
    rolling_std = train_df['value'].rolling(window=WINDOW_SIZE).std()

    # Use the last calculated values as our "model" for normal behavior
    # This is a naive approach but serves for the POC.
    NORMAL_MEAN = rolling_mean.iloc[-1]
    NORMAL_STD = rolling_std.iloc[-1]

    print(f"Training complete. Normal baseline: Mean={NORMAL_MEAN:.2f}, StdDev={NORMAL_STD:.2f}")

except FileNotFoundError:
    print(f"Error: Training data file not found at '{TRAIN_DATA_PATH}'")
    exit()


# --- Step 2: "Prediction" ---
# Load the test data and find anomalies.
# This whole section should be refactored into a reusable class/function.
try:
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
except FileNotFoundError:
    print(f"Error: Test data file not found at '{TEST_DATA_PATH}'")
    exit()

anomalies_found = []
print(f"\nScanning '{TEST_DATA_PATH}' for anomalies...")

# Inefficient row-by-row iteration
for index, row in test_df.iterrows():
    value = row['value']
    timestamp = row['timestamp']

    # Define upper and lower bounds for normal behavior
    upper_bound = NORMAL_MEAN + THRESHOLD_MULTIPLIER * NORMAL_STD
    lower_bound = NORMAL_MEAN - THRESHOLD_MULTIPLIER * NORMAL_STD

    print(NORMAL_MEAN)
    print(NORMAL_STD)
    print(upper_bound)
    print(lower_bound)

    # Check for anomaly
    if not (lower_bound <= value <= upper_bound):
        anomaly_details = {
            "timestamp": str(timestamp),
            "value": value,
            "reason": f"Value {value:.2f} is outside the normal range [{lower_bound:.2f}, {upper_bound:.2f}]"
        }
        anomalies_found.append(anomaly_details)
        print(f"  -> Anomaly Detected at {timestamp}: Value={value:.2f}")

# --- Step 3: Reporting ---
# Final summary of the results.
print("\n--- Scan Complete ---")
if anomalies_found:
    print(f"Total anomalies found: {len(anomalies_found)}")
    # In a real tool, this would be saved to a file, not just printed.
    # print("Details:", anomalies_found)
else:
    print("No anomalies were detected in the test data.")

### Some comments: 
### There are a few ways we can improve on this code 
### First: Write functions to load the data, check the loading was successful and handle any errors gracefully as well as log the process.
### Second: Write a script to conduct feature engineering on the dataset, such as creating additional features or aggregations.
### Third: Implement a more efficient anomaly detection algorithm, possibly using machine learning techniques.
### Fourth: Create a proper evaluation framework to assess the performance of the anomaly detection system.
### Fifth: Add the system into a fast api

### In terms of the model, this is a simple approach which I think is a good starting point. The main limitations are that is not incorporating any temporal aspects of the data other than 
### what is captured by the rolling window and that only the last window is used as the baseline for normal behavior for the test data. 

# Recommendations for Improving This Code:
# 1. Refactor data loading into dedicated functions that validate input, handle errors gracefully, and log the process for traceability.
# 2. Develop a feature engineering module to create additional features or aggregations, enhancing the model’s ability to detect anomalies.
# 3. Implement a more efficient and scalable anomaly detection algorithm, potentially leveraging machine learning techniques for improved accuracy.
# 4. Establish a robust evaluation framework to assess the performance of the anomaly detection system, including visualization and benchmarking.
# 5. Integrate the system into a FastAPI application to enable easy deployment and API-based access.
# 
# Note: The current approach is a simple proof-of-concept. 
# Its main limitations are the lack of advanced temporal modeling and features (beyond the rolling window) and reliance on the last window as the baseline for normal behavior.
# 
# - Make configuration parameters (paths, window size, thresholds) easily adjustable.
# - Modularize code for maintainability and extensibility.
# - Add options for saving and visualizing results.