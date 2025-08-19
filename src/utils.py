
### File to create functions used throughout the project
import numpy as np 
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.statsmodels
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
import mlflow.statsmodels
from sklearn.ensemble import RandomForestRegressor
import pickle
import joblib
import json
from datetime import datetime

### Logger setup
def setup_logger(log_path=None, log_level=logging.INFO):
    """
    Sets up logging with a specified log file path and log level.
    """
    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )


### Data loader incorporating logging and some basic information about the dataset. 
def load_data(file_path, log_path=None):
    """
    Loads a CSV file into a pandas DataFrame.
    Logs missing values, number of rows/columns, and column names. Handles errors.
    """
    if log_path:
        logging.basicConfig(filename=log_path, level=logging.INFO)
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path} successfully.")
        n_rows, n_cols = df.shape
        logging.info(f"Rows: {n_rows}, Columns: {n_cols}")
        logging.info(f"Column names: {list(df.columns)}")

        # Log missing values summary
        missing_summary = pd.DataFrame({
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
        })
        missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        logging.info("Missing values summary:")
        if missing_summary.empty:
            logging.info("No missing values in the dataset.")
        else:
            logging.info(f"\n{missing_summary}")

        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        print(f"Error loading {file_path}: {e}")
        return None
    

def plot_and_describe(df, timestamp_col='timestamp', columns=None):
    """
    Plots and describes specified columns in the DataFrame.
    For numeric columns, prints min/max. For datetime, prints min/max and step size.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        timestamp_col (str): Name of the timestamp column (default='timestamp').
        columns (list or None): List of columns to plot and describe. If None, uses all columns.
    """
    df = df.copy()
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    if columns is None:
        columns = df.columns
    for col in columns:
        if col not in df.columns:
            print(f"{col}: Column not found in DataFrame.")
            continue
        unique_count = df[col].nunique()
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"{col}: Unique values = {unique_count}, Range = [{min_val}, {max_val}]")
            plt.figure(figsize=(12, 4))
            plt.plot(df[timestamp_col], df[col], marker='.', linestyle='-', alpha=0.7)
            plt.xlabel(timestamp_col)
            plt.ylabel(col)
            plt.title(f'{col} Over Time')
            plt.tight_layout()
            plt.show()
        elif col == timestamp_col and pd.api.types.is_datetime64_any_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"{col}: Unique values = {unique_count}, Range = [{min_val}, {max_val}]")
            if len(df) > 1:
                step = (df[col].iloc[1] - df[col].iloc[0])
                print(f"Step between observations: {step}")
        else:
            print(f"{col}: Unique values = {unique_count}")

# Feature engineering functions 
# function to add features based on timestamp. I included a sine and cosine transform for the hour of the day since the data follows a strong temporal trend. 
def add_temporal_features(df, timestamp_col='timestamp'):
    """
    Adds temporal features to a DataFrame with a 'timestamp' column.
    Converts 'timestamp_col' to datetime, extracts hour, minute, dayofweek, and adds sine/cosine transforms for daily cycles.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing a 'timestamp' column.

    Returns:
        pd.DataFrame: DataFrame with new temporal features added.
    """
    if timestamp_col not in df.columns:
        logging.error("'timestamp' column not found in DataFrame.")
        raise KeyError("'timestamp' column not found in DataFrame.")
    df = df.copy()
    try:
        df['timestamp'] = pd.to_datetime(df[timestamp_col])
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        # Sine/cosine transforms for daily cycle
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        logging.info("Temporal features added successfully.")
    except Exception as e:
        logging.error(f"Error adding temporal features: {e}")
        raise
    return df

# This function helps add features based on the mean, std, min and max values on the rolling window. I start with a rolling window of an hour as default if no window is provided. 
def add_rolling_features(df, window=60, col='value'):
    """
    Adds rolling window statistics (mean, std, min, max) for the specified column.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the target column.
        window (int): Window size for rolling calculations (default=60).
        col (str): Name of the column to compute rolling statistics on (default='value').

    Returns:
        pd.DataFrame: DataFrame with new rolling statistics features added.
    """
    if col not in df.columns:
        logging.error(f"'{col}' column not found in DataFrame.")
        raise KeyError(f"'{col}' column not found in DataFrame.")
    df = df.copy()
    try:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=window, min_periods=1).std()
        df[f'{col}_rolling_min'] = df[col].rolling(window=window, min_periods=1).min()
        df[f'{col}_rolling_max'] = df[col].rolling(window=window, min_periods=1).max()
        logging.info(f"Rolling features added for column '{col}' with window={window}.")
    except Exception as e:
        logging.error(f"Error adding rolling features: {e}")
        raise
    return df

# This function will compute features based on the lagged values. By default I compute three features for the most recent three values.  
def add_lag_features(df, lags=[1, 2, 3], col='value'):
    """
    Adds lagged versions of the specified column to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the target column.
        lags (list): List of lag periods to create (default=[1, 2, 3]).
        col (str): Name of the column to create lag features for (default='value').

    Returns:
        pd.DataFrame: DataFrame with new lag features added.
    """
    if col not in df.columns:
        logging.error(f"'{col}' column not found in DataFrame.")
        raise KeyError(f"'{col}' column not found in DataFrame.")
    df = df.copy()
    try:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        logging.info(f"Lag features added for column '{col}' and lags: {lags}.")
    except Exception as e:
        logging.error(f"Error adding lag features: {e}")
        raise
    return df


# Functions to train the models. We start with the naive window rolling model 

def train_model_naive(train_df, window_size):
	"""
	Trains anomaly detection model by calculating rolling mean and std from training data.
	Returns the last rolling mean and std as baseline. This is in line with the naive implementation 
	"""
	train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
	train_df.set_index('timestamp', inplace=True)
	rolling_mean = train_df['value'].rolling(window=window_size).mean()
	rolling_std = train_df['value'].rolling(window=window_size).std()
	normal_mean = rolling_mean.iloc[-1]
	normal_std = rolling_std.iloc[-1]
	print(f"Training complete. Normal baseline: Mean={normal_mean:.2f}, StdDev={normal_std:.2f}")
	return normal_mean, normal_std

# Function to test the model and return the anomalies found on the validation or test set
def test_model_naive(test_df, normal_mean, normal_std, threshold_multiplier):
	"""
	Tests anomaly detection model on test data, returns list of anomalies found.
	"""
	test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
	anomalies_found = []
	upper_bound = normal_mean + threshold_multiplier * normal_std
	lower_bound = normal_mean - threshold_multiplier * normal_std
	print(f"\nScanning '{test_df}' for anomalies...")
	for index, row in test_df.iterrows():
		value = row['value']
		timestamp = row['timestamp']
		if not (lower_bound <= value <= upper_bound):
			anomaly_details = {
				"timestamp": str(timestamp),
				"value": value,
				"reason": f"Value {value:.2f} is outside the normal range [{lower_bound:.2f}, {upper_bound:.2f}]"
			}
			anomalies_found.append(anomaly_details)
			#print(f"  -> Anomaly Detected at {timestamp}: Value={value:.2f}")
	print("\n--- Scan Complete ---")
	if anomalies_found:
		print(f"Total anomalies found: {len(anomalies_found)}")
	else:
		print("No anomalies were detected in the test data.")
	return anomalies_found



# Functions to validate and compare across models 

# Function to plot validation data and highlight anomalies
def plot_validation_with_anomalies(validation_df, anomalies, value_col='value', timestamp_col='timestamp', save_path=None):
    """
    Plots the validation data and highlights anomalies in red.
    Prints statistics about the anomalies.
    Optionally saves the plot to the specified path.
    """
    # Convert anomalies to DataFrame for easy plotting
    if anomalies is not None and isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
        anomalies_df = anomalies.copy()
        anomalies_df[timestamp_col] = pd.to_datetime(anomalies_df[timestamp_col])
    elif anomalies is not None and not isinstance(anomalies, pd.DataFrame) and len(anomalies) > 0:
        anomalies_df = pd.DataFrame(anomalies)
        anomalies_df[timestamp_col] = pd.to_datetime(anomalies_df[timestamp_col])
    else:
        anomalies_df = pd.DataFrame(columns=[timestamp_col, value_col])

    plt.figure(figsize=(12, 6))
    plt.plot(validation_df[timestamp_col], validation_df[value_col], label='Validation Data', alpha=0.7)
    if not anomalies_df.empty:
        plt.scatter(anomalies_df[timestamp_col], anomalies_df[value_col], color='red', label='Anomalies', zorder=5)
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.title('Validation Data with Anomalies Highlighted')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

    # Print anomaly statistics
    print(f"Total anomalies: {len(anomalies)}")
    if not anomalies_df.empty:
        print(f"Anomaly value range: [{anomalies_df[value_col].min()}, {anomalies_df[value_col].max()}]")
        print(f"Anomaly timestamps: {anomalies_df[timestamp_col].min()} to {anomalies_df[timestamp_col].max()}")
        print(anomalies_df[[timestamp_col, value_col]].head())



# Function to plot validation data vs predictions
def plot_validation_vs_predictions(validation_df, y_pred, value_col='value', timestamp_col='timestamp', save_path=None):
	plt.figure(figsize=(12, 6))
	plt.plot(validation_df[timestamp_col], validation_df[value_col], label='Actual', alpha=0.7)
	plt.plot(validation_df[timestamp_col], y_pred, label='Predicted', alpha=0.7)
	plt.xlabel('Timestamp')
	plt.ylabel('Value')
	plt.title('Validation Data vs Predictions')
	plt.legend()
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path)
		print(f"Plot saved to {save_path}")
	plt.show()

# Model assesment 
# Function to assess anomaly separation from local neighbors
def comprehensive_anomaly_analysis(anomalies_df, validation_df, value_col='value', timestamp_col='timestamp', window=6):
    """
    Comprehensive anomaly analysis combining local separation metrics with global anomaly statistics.
    
    Parameters:
        anomalies_df (pd.DataFrame): DataFrame of detected anomalies
        validation_df (pd.DataFrame): Full validation dataset
        value_col (str): Name of the value column
        timestamp_col (str): Name of the timestamp column  
        window (int): Number of neighbors before and after to consider for local separation
        
    Returns:
        dict: Comprehensive metrics dictionary containing:
            - Basic counts and rates
            - Temporal spread metrics
            - Neighborhood comparison metrics
            - Local separation statistics
            - Individual anomaly details
    """
    
    if len(anomalies_df) == 0:
        return {
            'num_anomalies': 0,
            'anomaly_rate': 0.0,
            'mean_anomaly_magnitude': 0.0,
            'max_anomaly_magnitude': 0.0,
            'anomaly_temporal_spread': 0.0,
            'mean_neighborhood_diff': 0.0,
            'max_neighborhood_diff': 0.0,
            'mean_local_separation': 0.0,
            'max_local_separation': 0.0,
            'anomaly_details': pd.DataFrame()
        }
    
    # Basic counts
    num_anomalies = len(anomalies_df)
    anomaly_rate = num_anomalies / len(validation_df)
    
    # Initialize lists for metrics
    anomaly_magnitudes = []
    neighborhood_diffs = []
    local_separations = []
    anomaly_details = []
    
    # Process each anomaly
    for _, anomaly in anomalies_df.iterrows():
        anomaly_time = anomaly[timestamp_col]
        anomaly_value = anomaly[value_col]
        anomaly_idx = anomaly.name if anomaly.name is not None else _
        
        # === TIME-BASED NEIGHBORHOOD ANALYSIS (±30 minutes) ===
        time_window = pd.Timedelta(minutes=30)
        time_neighbors = validation_df[
            (validation_df[timestamp_col] >= anomaly_time - time_window) &
            (validation_df[timestamp_col] <= anomaly_time + time_window) &
            (validation_df[timestamp_col] != anomaly_time)
        ]
        
        time_neighbor_diff = 0.0
        if len(time_neighbors) > 0:
            neighbor_median = time_neighbors[value_col].median()
            time_neighbor_diff = abs(anomaly_value - neighbor_median)
            neighborhood_diffs.append(time_neighbor_diff)
        
        # === INDEX-BASED LOCAL SEPARATION ANALYSIS ===
        local_separation = np.nan
        local_mean = np.nan
        local_std = np.nan
        
        try:
            # Find the index position of this anomaly in validation_df
            if anomaly_idx in validation_df.index:
                val_idx_pos = validation_df.index.get_loc(anomaly_idx)
                start = max(0, val_idx_pos - window)
                end = min(len(validation_df), val_idx_pos + window + 1)
                local_data = validation_df[value_col].iloc[start:end]
                
                # Remove the anomaly point itself from local calculation
                local_data_clean = local_data.drop(validation_df.index[val_idx_pos], errors='ignore')
                
                if len(local_data_clean) > 0:
                    local_mean = local_data_clean.mean()
                    local_std = local_data_clean.std()
                    
                    if local_std > 0:
                        local_separation = abs(anomaly_value - local_mean) / local_std
                        local_separations.append(local_separation)
        except Exception as e:
            # If index-based approach fails, skip local separation for this anomaly
            pass
        
        # === GLOBAL MAGNITUDE ===
        global_mean = validation_df[value_col].mean()
        anomaly_magnitude = abs(anomaly_value - global_mean)
        anomaly_magnitudes.append(anomaly_magnitude)
        
        # Store detailed information for this anomaly
        anomaly_details.append({
            'index': anomaly_idx,
            'timestamp': anomaly_time,
            'value': anomaly_value,
            'global_magnitude': anomaly_magnitude,
            'time_neighbor_diff': time_neighbor_diff,
            'local_mean': local_mean,
            'local_std': local_std,
            'local_separation': local_separation
        })
    
    # === TEMPORAL SPREAD ANALYSIS ===
    temporal_spread = 0.0
    if len(anomalies_df) > 1:
        # Sort by timestamp to calculate proper time differences
        sorted_anomalies = anomalies_df.sort_values(timestamp_col)
        time_diffs = sorted_anomalies[timestamp_col].diff().dt.total_seconds() / 3600  # hours
        temporal_spread = time_diffs.std()
    
    # === COMPILE COMPREHENSIVE METRICS ===
    metrics = {
        # Basic counts
        'num_anomalies': num_anomalies,
        'anomaly_rate': anomaly_rate,
        
        # Global magnitude metrics
        'mean_anomaly_magnitude': np.mean(anomaly_magnitudes) if anomaly_magnitudes else 0.0,
        'max_anomaly_magnitude': np.max(anomaly_magnitudes) if anomaly_magnitudes else 0.0,
        
        # Time-based neighborhood metrics
        'mean_neighborhood_diff': np.mean(neighborhood_diffs) if neighborhood_diffs else 0.0,
        'max_neighborhood_diff': np.max(neighborhood_diffs) if neighborhood_diffs else 0.0,
        
        # Index-based local separation metrics
        'mean_local_separation': np.mean(local_separations) if local_separations else 0.0,
        'max_local_separation': np.max(local_separations) if local_separations else 0.0,
        'num_valid_separations': len(local_separations),
        
        # Temporal distribution
        'anomaly_temporal_spread': temporal_spread,
        
        # Detailed breakdown
        'anomaly_details': pd.DataFrame(anomaly_details)
    }
    
    return metrics

# Log metrics in mlflow
def log_comprehensive_model_metrics(model_name, y_true, y_pred, anomalies_df, validation_df, 
                                   value_col='value', timestamp_col='timestamp', 
                                   model_params=None, model_object=None, window=6):
    """
    Enhanced MLflow logging using comprehensive anomaly analysis.
    
    Parameters:
        model_name (str): Name of the model for the MLflow run
        y_true (array-like): True values (for models with predictions, can be None for pure anomaly detection)
        y_pred (array-like): Predicted values (can be None for pure anomaly detection)
        anomalies_df (pd.DataFrame): DataFrame containing detected anomalies
        validation_df (pd.DataFrame): Full validation dataset
        value_col (str): Name of the value column
        timestamp_col (str): Name of the timestamp column
        model_params (dict): Model parameters to log
        model_object: Model object to save (sklearn, statsmodels, etc.)
        window (int): Window size for local separation analysis
    """

    
    with mlflow.start_run(run_name=model_name):
        
        # === MODEL PERFORMANCE METRICS (if predictions available) ===
        if y_pred is not None and y_true is not None:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mape", mape)
            
            # Residual analysis
            residuals = y_true - y_pred
            mlflow.log_metric("residual_mean", residuals.mean())
            mlflow.log_metric("residual_std", residuals.std())
            mlflow.log_metric("residual_skew", residuals.skew())
            mlflow.log_metric("residual_kurtosis", residuals.kurtosis())
        else:
            rmse = mae = mape = None
            residuals = None
        
        # === COMPREHENSIVE ANOMALY ANALYSIS ===
        anomaly_metrics = comprehensive_anomaly_analysis(
            anomalies_df, validation_df, 
            value_col=value_col, 
            timestamp_col=timestamp_col, 
            window=window
        )
        
        # Log all anomaly metrics with prefixes for organization
        basic_metrics = ['num_anomalies', 'anomaly_rate']
        magnitude_metrics = ['mean_anomaly_magnitude', 'max_anomaly_magnitude'] 
        neighborhood_metrics = ['mean_neighborhood_diff', 'max_neighborhood_diff']
        separation_metrics = ['mean_local_separation', 'max_local_separation', 'num_valid_separations']
        temporal_metrics = ['anomaly_temporal_spread']
        
        for metric in basic_metrics:
            mlflow.log_metric(f"basic_{metric}", anomaly_metrics[metric])
            
        for metric in magnitude_metrics:
            mlflow.log_metric(f"magnitude_{metric}", anomaly_metrics[metric])
            
        for metric in neighborhood_metrics:
            mlflow.log_metric(f"neighborhood_{metric}", anomaly_metrics[metric])
            
        for metric in separation_metrics:
            mlflow.log_metric(f"separation_{metric}", anomaly_metrics[metric])
            
        for metric in temporal_metrics:
            mlflow.log_metric(f"temporal_{metric}", anomaly_metrics[metric])
        
        # === LOG MODEL PARAMETERS ===
        if model_params:
            mlflow.log_params(model_params)
        
        # Add analysis parameters
        mlflow.log_param("separation_window", window)
        mlflow.log_param("neighborhood_window_minutes", 30)
        
        # === LOG MODEL OBJECT ===
        if model_object is not None:
            try:
                if hasattr(model_object, 'fit'):  # sklearn-like models
                    mlflow.sklearn.log_model(model_object, f"{model_name}_model")
                elif hasattr(model_object, 'summary'):  # statsmodels
                    mlflow.statsmodels.log_model(model_object, f"{model_name}_model")
            except Exception as e:
                mlflow.log_param("model_save_error", str(e))
        
        # === ENHANCED VISUALIZATION ===
        if y_pred is not None and y_true is not None:
            # For models with predictions - 6 subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Subplot 1: Predictions vs Actual
            axes[0,0].scatter(y_true, y_pred, alpha=0.5)
            axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            axes[0,0].set_xlabel('Actual')
            axes[0,0].set_ylabel('Predicted')
            axes[0,0].set_title(f'{model_name}: Predictions vs Actual')
            
            # Subplot 2: Residuals
            axes[0,1].scatter(y_pred, residuals, alpha=0.5)
            axes[0,1].axhline(y=0, color='r', linestyle='--')
            axes[0,1].set_xlabel('Predicted')
            axes[0,1].set_ylabel('Residuals')
            axes[0,1].set_title(f'{model_name}: Residual Plot')
            
            # Subplot 3: Residual distribution
            axes[0,2].hist(residuals, bins=30, alpha=0.7)
            axes[0,2].set_xlabel('Residuals')
            axes[0,2].set_ylabel('Frequency')
            axes[0,2].set_title(f'{model_name}: Residual Distribution')
        else:
            # For pure anomaly detection - 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Subplot 4: Anomaly values distribution
        ax_idx = (1,0) if y_pred is not None else (0,0)
        if len(anomalies_df) > 0:
            anomaly_values = anomalies_df[value_col].values
            axes[ax_idx].hist(anomaly_values, bins=20, alpha=0.7, label='Anomalies')
            axes[ax_idx].axvline(validation_df[value_col].mean(), color='r', linestyle='--', label='Overall Mean')
            axes[ax_idx].set_xlabel('Value')
            axes[ax_idx].set_ylabel('Frequency')
            axes[ax_idx].set_title(f'{model_name}: Anomaly Value Distribution')
            axes[ax_idx].legend()
        else:
            axes[ax_idx].text(0.5, 0.5, 'No Anomalies Detected', ha='center', va='center', transform=axes[ax_idx].transAxes)
            axes[ax_idx].set_title(f'{model_name}: No Anomalies')
        
        # Subplot 5: Local separation scores
        ax_idx = (1,1) if y_pred is not None else (0,1)
        details_df = anomaly_metrics['anomaly_details']
        if len(details_df) > 0 and 'local_separation' in details_df.columns:
            valid_separations = details_df['local_separation'].dropna()
            if len(valid_separations) > 0:
                axes[ax_idx].hist(valid_separations, bins=15, alpha=0.7, color='orange')
                axes[ax_idx].axvline(valid_separations.mean(), color='r', linestyle='--', label=f'Mean: {valid_separations.mean():.2f}')
                axes[ax_idx].set_xlabel('Local Separation (std devs)')
                axes[ax_idx].set_ylabel('Frequency')
                axes[ax_idx].set_title(f'{model_name}: Local Separation Scores')
                axes[ax_idx].legend()
            else:
                axes[ax_idx].text(0.5, 0.5, 'No Valid Separations', ha='center', va='center', transform=axes[ax_idx].transAxes)
                axes[ax_idx].set_title(f'{model_name}: No Separation Data')
        else:
            axes[ax_idx].text(0.5, 0.5, 'No Separation Data', ha='center', va='center', transform=axes[ax_idx].transAxes)
            axes[ax_idx].set_title(f'{model_name}: No Separation Data')
        
        # Subplot 6: Neighborhood differences (only for models with predictions)
        if y_pred is not None:
            if len(details_df) > 0 and 'time_neighbor_diff' in details_df.columns:
                neighbor_diffs = details_df['time_neighbor_diff']
                neighbor_diffs = neighbor_diffs[neighbor_diffs > 0]  # Remove zeros
                if len(neighbor_diffs) > 0:
                    axes[1,2].hist(neighbor_diffs, bins=15, alpha=0.7, color='green')
                    axes[1,2].axvline(neighbor_diffs.mean(), color='r', linestyle='--', label=f'Mean: {neighbor_diffs.mean():.3f}')
                    axes[1,2].set_xlabel('Difference from Time Neighbors')
                    axes[1,2].set_ylabel('Frequency')
                    axes[1,2].set_title(f'{model_name}: Neighborhood Differences')
                    axes[1,2].legend()
                else:
                    axes[1,2].text(0.5, 0.5, 'No Neighborhood Data', ha='center', va='center', transform=axes[1,2].transAxes)
                    axes[1,2].set_title(f'{model_name}: No Neighborhood Data')
            else:
                axes[1,2].text(0.5, 0.5, 'No Neighborhood Data', ha='center', va='center', transform=axes[1,2].transAxes)
                axes[1,2].set_title(f'{model_name}: No Neighborhood Data')
        
        plt.tight_layout()
        mlflow.log_figure(fig, f"{model_name}_comprehensive_analysis.png")
        plt.close()
        
        # === LOG DETAILED ANOMALY DATA ===
        if len(details_df) > 0:
            # Save anomaly details as CSV artifact
            details_csv = f"{model_name}_anomaly_details.csv"
            details_df.to_csv(details_csv, index=False)
            mlflow.log_artifact(details_csv)
            
            # Log summary statistics of the details
            mlflow.log_metric("avg_local_separation", details_df['local_separation'].mean())
            mlflow.log_metric("median_local_separation", details_df['local_separation'].median())
            mlflow.log_metric("avg_neighbor_diff", details_df['time_neighbor_diff'].mean())
            mlflow.log_metric("median_neighbor_diff", details_df['time_neighbor_diff'].median())
        
        # === SUMMARY PRINT ===
        if y_pred is not None:
            print(f"✅ {model_name} - RMSE: {rmse:.4f}, Anomalies: {anomaly_metrics['num_anomalies']}")
        else:
            print(f"✅ {model_name} - Anomalies: {anomaly_metrics['num_anomalies']}")
        
        print(f"   📊 Local Sep: {anomaly_metrics['mean_local_separation']:.2f}±{anomaly_metrics['max_local_separation']:.2f}")
        print(f"   🏘️  Neighbor Diff: {anomaly_metrics['mean_neighborhood_diff']:.4f}")
        print(f"   ⏰ Temporal Spread: {anomaly_metrics['anomaly_temporal_spread']:.2f}h")



# Function for random forest training and testing 
# Random Forest Model Training and Evaluation
def train_and_evaluate_random_forest(train_df, test_df, feature_cols, 
                                    n_estimators=100, random_state=42, 
                                    anomaly_threshold_std=2.5, 
                                    value_col='value', timestamp_col='timestamp'):
    """Train Random Forest model and evaluate on test data with anomaly detection."""
    
    # Check available features
    available_cols = [col for col in feature_cols if col in train_df.columns]
    if not available_cols:
        raise ValueError(f"None of the specified features {feature_cols} found in training data")
    
    print(f"Using features: {available_cols}")
    
    # Prepare training data
    X_train = train_df[available_cols].dropna()
    y_train = train_df.loc[X_train.index, value_col]
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Get training residuals to establish threshold (same as API approach)
    y_train_pred = rf_model.predict(X_train)
    train_residuals = y_train - y_train_pred
    train_residual_std = train_residuals.std()
    
    print(f"Training residual std: {train_residual_std:.4f}")
    print(f"Anomaly threshold: {anomaly_threshold_std:.1f} * {train_residual_std:.4f} = {anomaly_threshold_std * train_residual_std:.4f}")
    
    # Predict on test data
    X_test = test_df[available_cols].dropna()
    y_test = test_df.loc[X_test.index, value_col]
    y_pred = rf_model.predict(X_test)
    
    # Anomaly detection using TRAINING residual std (consistent with API)
    test_residuals = y_test - y_pred
    anomaly_mask = np.abs(test_residuals) > (anomaly_threshold_std * train_residual_std)
    anomalies_df = test_df.loc[X_test.index][anomaly_mask]
    
    print(f"Anomalies (Random Forest): {len(anomalies_df)} using training-based threshold")
    
    # Log Random Forest to MLflow
    rf_params = {
        'n_estimators': n_estimators,
        'features': available_cols,
        'model_type': 'RandomForestRegressor',
        'random_state': random_state,
        'anomaly_threshold_std': anomaly_threshold_std,
        'train_residual_std': train_residual_std
    }
    
    log_comprehensive_model_metrics("Random_Forest", y_test, y_pred, anomalies_df, test_df,
                                   value_col=value_col, timestamp_col=timestamp_col,
                                   model_params=rf_params, model_object=rf_model)
    
    if len(anomalies_df) > 0:
        plot_validation_with_anomalies(test_df, anomalies_df, 
                                      save_path="results/test_anomalies_rf.png")
    
    return rf_model, y_pred, anomalies_df


### save and load functions for the final model 

def save_model_for_api(model, model_name, feature_cols, model_params, training_stats=None):
    """
    Save trained model and metadata for FastAPI deployment.
    
    Parameters:
        model: Trained model object
        model_name: Name of the model (e.g., 'linear_regression', 'random_forest')
        feature_cols: List of feature columns used
        model_params: Model parameters dictionary
        training_stats: Optional training statistics
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Save the model using pickle (most compatible)
        model_path = f"{model_dir}/{model_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Also save with joblib as backup (better for sklearn models)
        joblib_path = f"{model_dir}/{model_name}_model.joblib"
        joblib.dump(model, joblib_path)
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'model_type': model_params.get('model_type', 'Unknown'),
            'feature_cols': feature_cols,
            'model_params': model_params,
            'training_timestamp': timestamp,
            'training_stats': training_stats,
            'model_path': model_path,
            'joblib_path': joblib_path,
            'python_version': str(pd.__version__),  # Use pandas version as proxy
        }
        
        metadata_path = f"{model_dir}/{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save feature columns separately for easy access
        features_path = f"{model_dir}/{model_name}_features.json"
        with open(features_path, 'w') as f:
            json.dump({'features': feature_cols}, f, indent=2)
        
        print(f"✅ Model saved successfully:")
        print(f"   📁 Directory: {model_dir}")
        print(f"   🤖 Model: {model_path}")
        print(f"   📋 Metadata: {metadata_path}")
        print(f"   🏷️  Features: {features_path}")
        
        return {
            'model_dir': model_dir,
            'model_path': model_path,
            'metadata_path': metadata_path,
            'features_path': features_path
        }
        
    except Exception as e:
        print(f"❌ Failed to save model {model_name}: {e}")
        return None

def load_model_for_api(model_name):
    """
    Load a saved model for FastAPI deployment.
    
    Parameters:
        model_name: Name of the model to load
        
    Returns:
        dict: Dictionary containing model, metadata, and features
    """
    model_dir = f"models/{model_name}"
    
    try:
        # Load model
        model_path = f"{model_dir}/{model_name}_model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_path = f"{model_dir}/{model_name}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load features
        features_path = f"{model_dir}/{model_name}_features.json"
        with open(features_path, 'r') as f:
            features = json.load(f)
        
        print(f"✅ Model loaded successfully: {model_name}")
        
        return {
            'model': model,
            'metadata': metadata,
            'features': features['features'],
            'model_type': metadata.get('model_type', 'Unknown')
        }
        
    except Exception as e:
        print(f"❌ Failed to load model {model_name}: {e}")
        return None
