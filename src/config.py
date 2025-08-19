### config file for the project. 
#   
# File paths
TRAIN_DATA_PATH = "data/sensor_data_train.csv"
TEST_DATA_PATH = "data/sensor_data_test.csv"
FEATURED_TRAIN_PATH = "data/featured_sensor_data_train.csv"
FEATURED_TEST_PATH = "data/featured_sensor_data_test.csv"

LOG_EDA_PATH = "logs/anomaly_detection_eda.log"
LOG_FEATURE_PATH = "logs/anomaly_detection_feature.log"
LOG_MODEL_SELECTION_PATH = "logs/model_selection.log"
MODEL_PATH = "model/anomaly_model.joblib"
FEATURES_PATH = "model/features.txt"
LOG_MODEL_TRAINING_PATH = "logs/model_training.log"
MODEL_SAVE_PATH = "models/random_forest"

# Parameters for rolling window model 
WINDOW_SIZE = 60  # Size of rolling window for statistics
THRESHOLD_MULTIPLIER = 3.0  # Number of std deviations for anomaly threshold

# Model parameters (random forest model)
N_ESTIMATORS = 100
MAX_DEPTH = None
RANDOM_STATE = 42
rf_feature_cols = ['hour', 'minute']
anomaly_threshold_std=3

# Other configuration
SAVE_RESULTS_PATH = "results/anomaly_results.csv"
