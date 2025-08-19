import pandas as pd
import logging
import os

# Example config imports
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, FEATURED_TRAIN_PATH, FEATURED_TEST_PATH, LOG_FEATURE_PATH
from utils import setup_logger, load_data, add_temporal_features, add_rolling_features, add_lag_features

# Function to save the transformed data 
def save_featured_data(df, path):
    df.to_csv(path, index=False)

# Apply all the transformations for feature engineering  
def process_and_log(data_path, featured_path, log_path, timestamp_col='timestamp', value_col='value'):
    setup_logger(log_path=log_path)
    df = load_data(data_path, log_path=log_path)
    if df is None:
        logging.error(f"Failed to load data from {data_path}. Skipping feature engineering.")
        return
    logging.info(f"Loaded data from {data_path}: {df.shape}")
    df = add_temporal_features(df, timestamp_col=timestamp_col)
    logging.info("Added temporal features.")
    df = add_rolling_features(df, window=60, col=value_col)
    logging.info("Added rolling features.")
    df = add_lag_features(df, lags=[1, 2, 3], col=value_col)
    logging.info("Added lag features.")
    save_featured_data(df, featured_path)
    logging.info(f"Saved featured data to {featured_path}.")


# Caller process for both the train and test data
def main():
    # Train data
    process_and_log(
        data_path=TRAIN_DATA_PATH,
        featured_path=FEATURED_TRAIN_PATH,
        log_path=LOG_FEATURE_PATH,
        timestamp_col='timestamp',
        value_col='value'
    )
    # Test data
    process_and_log(
        data_path=TEST_DATA_PATH,
        featured_path=FEATURED_TEST_PATH,
        log_path=LOG_FEATURE_PATH,
        timestamp_col='timestamp',
        value_col='value'
    )
    logging.info("Feature engineering pipeline completed for train and test data.")

if __name__ == "__main__":
    main()