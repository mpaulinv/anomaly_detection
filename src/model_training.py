# Model training and testing 

import numpy as np 
import pandas as pd
import os
import logging
import io 
import matplotlib.pyplot as plt
import pickle
import joblib
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import mlflow
import mlflow.sklearn
import mlflow.statsmodels
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor

# Import feature engineering pipeline
from feature_engineering import process_and_log

# Model parameters (random forest model). I use a more strict anomaly threshold than in training as I want to flag all significant deviations. 
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, LOG_MODEL_TRAINING_PATH, rf_feature_cols, N_ESTIMATORS, RANDOM_STATE, anomaly_threshold_std
from utils import setup_logger, load_data, train_and_evaluate_random_forest, log_comprehensive_model_metrics, plot_validation_with_anomalies, save_model_for_api


def main(model_type='random_forest'):
    """
    Main training pipeline that processes raw data through feature engineering
    and trains the final model
    
    Args:
        model_type: Type of model to train ('random_forest', etc.)
        
    Returns:
        tuple: (model, predictions, anomalies) or (None, None, None) if failed
    """
    # Setup logging
    setup_logger(log_path=LOG_MODEL_TRAINING_PATH)
    logging.info("Starting model training pipeline...")
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    try:
        # Process raw data through feature engineering pipeline
        logging.info("Processing training data through feature engineering...")
        process_and_log(
            data_path=TRAIN_DATA_PATH,
            featured_path="data/featured_train_temp.csv",  # Temporary file
            log_path=LOG_MODEL_TRAINING_PATH,
            timestamp_col='timestamp',
            value_col='value'
        )
        
        logging.info("Processing test data through feature engineering...")
        process_and_log(
            data_path=TEST_DATA_PATH,
            featured_path="data/featured_test_temp.csv",  # Temporary file
            log_path=LOG_MODEL_TRAINING_PATH,
            timestamp_col='timestamp',
            value_col='value'
        )
        
        # Load the processed data
        train_features = load_data("data/featured_train_temp.csv", log_path=LOG_MODEL_TRAINING_PATH)
        test_features = load_data("data/featured_test_temp.csv", log_path=LOG_MODEL_TRAINING_PATH)
        
        # Data validation
        if train_features is None or test_features is None:
            logging.error("Failed to load processed training or test data")
            return None, None, None
            
        if train_features.empty or test_features.empty:
            logging.error("Training or test data is empty after feature engineering")
            return None, None, None
        
        logging.info(f"Training data shape: {train_features.shape}")
        logging.info(f"Test data shape: {test_features.shape}")
        
        # Get actual feature columns from processed data (exclude timestamp and value)
        exclude_cols = {'timestamp', 'value'}
        actual_feature_cols = [col for col in train_features.columns if col not in exclude_cols]
        logging.info(f"Available features: {actual_feature_cols}")
        
        # Use actual features or configured features
        feature_cols_to_use = rf_feature_cols if all(col in actual_feature_cols for col in rf_feature_cols) else actual_feature_cols
        logging.info(f"Using features: {feature_cols_to_use}")
        
        # Ensure timestamp is datetime
        train_features['timestamp'] = pd.to_datetime(train_features['timestamp'])
        test_features['timestamp'] = pd.to_datetime(test_features['timestamp'])
        
        if model_type == 'random_forest':
            logging.info("Training Random Forest model...")
            
            # Train the model
            rf_model, rf_predictions, rf_anomalies = train_and_evaluate_random_forest(
                train_df=train_features,
                test_df=test_features, 
                feature_cols=feature_cols_to_use,
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
                anomaly_threshold_std=anomaly_threshold_std
            )
            
            # Calculate residual standard deviation from TRAINING set for API threshold
            # This matches the approach used in train_and_evaluate_random_forest
            X_train = train_features[feature_cols_to_use].dropna()
            y_train = train_features.loc[X_train.index, 'value']
            y_train_pred = rf_model.predict(X_train)
            train_residuals = y_train - y_train_pred
            residual_std = np.std(train_residuals)
            logging.info(f"Calculated training residual standard deviation: {residual_std:.4f}")
            logging.info(f"This will be used as the anomaly threshold base in the API")
            
            # Save the Random Forest model for API
            rf_params = {
                'model_type': 'RandomForestRegressor',
                'n_estimators': N_ESTIMATORS,
                'random_state': RANDOM_STATE,
                'anomaly_threshold_std': anomaly_threshold_std,
                'residual_std': residual_std,
                'features': feature_cols_to_use
            }
            
            # Log comprehensive metrics (this will start its own MLflow run)
            log_comprehensive_model_metrics(
                model_name='RandomForest_Final',
                y_true=test_features['value'].values,
                y_pred=rf_predictions,
                anomalies_df=rf_anomalies,
                validation_df=test_features,
                value_col='value',
                timestamp_col='timestamp',
                model_params=rf_params,
                model_object=rf_model
            )
            
            rf_training_stats = {
                'anomaly_count': len(rf_anomalies),
                'training_samples': len(train_features),
                'test_samples': len(test_features),
                'total_features': len(feature_cols_to_use),
                'residual_std': residual_std,
                'residual_mean': np.mean(train_residuals),
                'residual_min': np.min(train_residuals),
                'residual_max': np.max(train_residuals)
            }
            
            save_model_for_api(
                model=rf_model,
                model_name='random_forest',
                feature_cols=feature_cols_to_use,
                model_params=rf_params,
                training_stats=rf_training_stats
            )
            
            logging.info("Model training completed successfully!")
            return rf_model, rf_predictions, rf_anomalies
        
        else:
            logging.error(f"Unsupported model type: {model_type}")
            return None, None, None
            
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        return None, None, None


os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)  

if __name__ == "__main__":
    # Run the main training pipeline
    model, predictions, anomalies = main(model_type='random_forest')
    
    if model is not None:
        print("Model training completed successfully!")
        print(f"Anomalies detected: {len(anomalies) if anomalies is not None else 0}")
    else:
        print("Model training failed!") 