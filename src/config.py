
### config. Elaborado por Mario Paulín como parte del proyecto agritech  
# Rutas de archivos
DATA_RAW_PATH = "data/yield_df.csv"
DATA_CLEAN_PATH = "data/maize_cleaned.csv"
LOG_PATH = "logs/data_pipeline.log"
KAGGLE_DATASET = "patelris/crop-yield-prediction-dataset"
MODEL_PATH = "model/model_rf_final.joblib"
FEATURES_PATH = "model/features_rf_area.txt"
DATA_PATH = "data/feature_engineered_data.csv"
DATA_CLEAN_PATH = "data/maize_cleaned.csv"
FEATURED_DATA_PATH = "data/feature_engineered_data.csv"
LOG_FEATURE_PATH = "logs/feature_engineering_pipeline.log"

# Columnas 
CONTINUOUS_COLUMNS = ['rain_fall_mean', 'pesticides_mean', 'avg_temp_median']

# Parámetros de BoxCox
LAMBDA_RAIN = 0.46461002838902316
LAMBDA_PEST = 0.09023987209681697
LAMBDA_TEMP = 1.2687660894233703

# Parámetros del modelo Random Forest
N_ESTIMATORS = 550
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = 2
RANDOM_STATE = 42