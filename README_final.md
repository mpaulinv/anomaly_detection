# Factory OS Anomaly Detection System

A production-ready anomaly detection system for industrial sensor data using machine learning. This system provides real-time anomaly detection through a REST API and comprehensive model training pipeline with experiment tracking.

## Project Overview

This project implements an end-to-end anomaly detection system for time-series sensor data. The system identifies unusual patterns in industrial sensor readings using Random Forest regression with statistical thresholds. Key features include:

- **Real-time anomaly detection API** built with FastAPI
- **Multiple model comparison** including Random Forest, Linear Regression, Isolation Forest, and temporal-aware approaches
- **Statistical consistency** between training and inference using residual standard deviation thresholds
- **Comprehensive experiment tracking** with MLflow integration
- **Production-ready deployment** with model versioning and metadata persistence

## Dataset Description

The dataset contains sensor readings from a single temperature sensor (TEMP-01) with the following characteristics:

- **Time Range**: August 1-21, 2025 (21 days)
- **Frequency**: 1-minute intervals (30,240 total observations)
- **Value Range**: 64.4°C to 75.61°C
- **Pattern**: Strong daily cyclical behavior with sinusoidal temperature patterns
- **Sensor ID**: TEMP-01 (single sensor deployment)

The data exhibits pronounced temporal patterns with values typically starting low in the morning, rising towards midday, and decreasing in the afternoon, making time-series analysis techniques highly appropriate.

## Results Visualization

The following chart shows the validation data with detected anomalies highlighted in red:

![Validation Data with Anomalies](results/validation_anomalies.png)

The visualization demonstrates:
- **Normal patterns**: Clear daily sinusoidal temperature cycles
- **Detected anomalies**: Red dots indicating significant deviations from expected patterns
- **Model effectiveness**: The 2.5x residual standard deviation threshold successfully captures genuine outliers
- **Temporal distribution**: Anomalies detected across different time periods, showing model robustness

## Architecture

### System Components

`
├── data/                          # Raw and processed datasets
├── src/                          # Core application code
│   ├── API.py                   # FastAPI application for real-time inference
│   ├── model_training.py        # End-to-end training pipeline
│   ├── feature_engineering.py   # Feature extraction pipeline
│   ├── utils.py                 # Shared utility functions
│   └── config.py               # Configuration management
├── scripts/                     # Analysis and exploration scripts
│   ├── EDA.py                  # Exploratory data analysis
│   └── model_selection.py      # Model comparison and selection
├── models/                      # Trained model artifacts
├── logs/                       # Application and training logs
└── results/                    # Visualization outputs
`

### Data Flow

1. **Raw Data** → **Feature Engineering** → **Model Training** → **Model Persistence**
2. **API Request** → **Feature Extraction** → **Model Prediction** → **Anomaly Detection** → **Response**

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- Git for version control

### Dependencies

Create a requirements.txt file by running:

`ash
pip freeze > requirements.txt
`

Key dependencies include:
- fastapi: Web framework for API development
- uvicorn: ASGI server for FastAPI
- scikit-learn: Machine learning algorithms
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- mlflow: Experiment tracking and model versioning
- matplotlib: Data visualization
- pydantic: Data validation for API

### Installation Steps

1. **Clone the repository**:
`ash
git clone <repository-url>
cd factory_os
`

2. **Create and activate virtual environment**:
`ash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
`

3. **Install dependencies**:
`ash
pip install -r requirements.txt
`

4. **Create necessary directories**:
`ash
mkdir data models logs results
`

## Usage

### 1. Feature Engineering

Process raw sensor data to extract temporal and statistical features:

`ash
python src/feature_engineering.py
`

**Generated Features**:
- **Temporal**: hour, minute, dayofweek, sin_hour, cos_hour
- **Rolling Statistics**: value_rolling_mean, value_rolling_std, value_rolling_min, value_rolling_max (60-minute window)
- **Lag Features**: value_lag_1, value_lag_2, value_lag_3

### 2. Model Training

Train the anomaly detection model with comprehensive evaluation:

`ash
python src/model_training.py
`

**Training Process**:
- Processes raw data through feature engineering pipeline
- Trains Random Forest model with 100 estimators
- Calculates training residual standard deviation for threshold setting
- Evaluates on test set using consistent statistical thresholds
- Logs comprehensive metrics to MLflow
- Saves model artifacts for API deployment

### 3. Model Comparison and Selection

Explore different anomaly detection approaches:

`ash
python scripts/model_selection.py
`

**Available Models**:
- **Linear Regression**: Interpretable model using temporal features
- **Random Forest**: Non-linear model capturing complex patterns  
- **Isolation Forest**: Unsupervised outlier detection
- **Adaptive Rolling Window**: Time-aware statistical approach
- **Temporal-Aware Online**: Streaming anomaly detection

### 4. API Deployment

Start the FastAPI server for real-time anomaly detection:

`ash
python src/API.py
`

**API Endpoints**:
- GET /: Health check and model status
- GET /model/info: Model metadata and configuration
- POST /predict: Single observation anomaly detection
- POST /predict/batch: Batch prediction for multiple observations

**API Documentation**: http://localhost:8000/docs

### 5. Experiment Tracking

Launch MLflow UI to view experiment results:

`ash
mlflow ui
`

**MLflow Dashboard**: http://localhost:5000

## API Usage Examples

### Single Prediction

`ash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "timestamp": "2025-08-22T14:30:00Z",
       "sensor_id": "TEMP-01",
       "value": 72.5
     }'
`

### Batch Prediction

`ash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '[
       {
         "timestamp": "2025-08-22T14:30:00Z",
         "sensor_id": "TEMP-01", 
         "value": 72.5
       },
       {
         "timestamp": "2025-08-22T14:31:00Z",
         "sensor_id": "TEMP-01",
         "value": 85.2
       }
     ]'
`

### Response Format

`json
{
  "timestamp": "2025-08-22T14:30:00Z",
  "sensor_id": "TEMP-01",
  "value": 72.5,
  "predicted_value": 71.8,
  "is_anomaly": false,
  "anomaly_score": 0.7,
  "message": "Normal observation"
}
`

## Model Details

### Random Forest Configuration

The final production model uses a Random Forest Regressor with the following configuration:

`python
RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
`

**Features Used**:
- hour: Hour of day (0-23)
- minute: Minute of hour (0-59)

### Anomaly Detection Methodology

The system uses a **statistical threshold approach** based on training residuals:

1. **Training Phase**: Calculate residual standard deviation from training predictions
2. **Threshold Setting**: Anomaly threshold = 2.5 × training_residual_std  
3. **Detection Logic**: |actual_value - predicted_value| > threshold
4. **Consistency**: Same threshold methodology used in both training evaluation and API inference

This approach ensures **statistical consistency** between training and production environments.

## Model Performance and Results

### Model Comparison Results

Based on comprehensive evaluation across multiple models:

| Model | Anomalies Detected | Anomaly Rate | Mean Local Separation | Performance |
|-------|-------------------|--------------|---------------------|-------------|
| **Random Forest** | 62-81 | 1.0-1.5% | 2.5+ std devs | **Selected** |
| Linear Regression | 65-75 | 1.2-1.4% | 2.2+ std devs | Good |
| Adaptive Rolling | 70-85 | 1.3-1.6% | 1.8+ std devs | Fair |
| Isolation Forest | 35-45 | 0.6-0.8% | 1.2+ std devs | Poor |
| Temporal-Aware | 78-92 | 1.4-1.7% | 2.1+ std devs | Good |

### Selection Criteria

**Random Forest was selected** based on:

1. **Performance**: High local separation scores and consistent anomaly detection
2. **Simplicity**: Uses only 2 features (hour, minute) for easy production deployment
3. **Flexibility**: Captures non-linear patterns without assuming functional form
4. **Robustness**: Can be easily retrained if data patterns change
5. **Production-Ready**: Simple feature requirements enable real-time inference

### Evaluation Metrics

The system uses comprehensive anomaly analysis including:

- **Local Separation**: Distance from anomalies to their 6 nearest neighbors
- **Temporal Separation**: Distance from anomalies to neighbors within 30-minute windows  
- **Neighborhood Comparison**: Statistical differences from local temporal context
- **Temporal Spread**: Distribution of anomalies across time periods

## Configuration

### Key Configuration Parameters (config.py)

`python
# Model Parameters
N_ESTIMATORS = 100
RANDOM_STATE = 42
anomaly_threshold_std = 2.5  # Statistical threshold multiplier

# Feature Engineering
rf_feature_cols = ['hour', 'minute']
WINDOW_SIZE = 60  # Rolling window size (minutes)

# Paths
TRAIN_DATA_PATH = "data/sensor_data_train.csv"
TEST_DATA_PATH = "data/sensor_data_test.csv"
MODEL_SAVE_PATH = "models/random_forest"
`

### Customization Options

- **Sensitivity Adjustment**: Modify anomaly_threshold_std (2.5 = moderate, 2.0 = high sensitivity, 3.0 = low sensitivity)
- **Feature Selection**: Add additional features in rf_feature_cols  
- **Model Parameters**: Adjust N_ESTIMATORS for model complexity vs. speed tradeoff
- **Rolling Window**: Change WINDOW_SIZE for different temporal context

## Experiment Tracking

### MLflow Integration

All experiments are automatically tracked with MLflow including:

- **Model Parameters**: Feature lists, hyperparameters, thresholds
- **Performance Metrics**: RMSE, MAE, MAPE for regression models
- **Anomaly Metrics**: Detection rates, separation scores, temporal spreads
- **Visualizations**: Residual plots, anomaly distributions, validation curves
- **Model Artifacts**: Serialized models, metadata, feature configurations

### Accessing Results

1. **Start MLflow UI**: mlflow ui
2. **Navigate to**: http://localhost:5000
3. **Compare experiments** across different model types and configurations
4. **Download artifacts** including model files and detailed analysis reports

## Production Deployment

### Model Persistence

Trained models are saved with complete metadata:

`
models/random_forest/
├── random_forest_model.pkl      # Pickled model object
├── random_forest_model.joblib   # Joblib backup
├── random_forest_metadata.json  # Training metadata & parameters  
└── random_forest_features.json  # Feature configuration
`

### API Deployment Considerations

1. **Health Monitoring**: Use / endpoint for health checks
2. **Model Validation**: Check /model/info for loaded model status
3. **Error Handling**: API provides detailed error messages for debugging
4. **Batch Processing**: Use /predict/batch for high-throughput scenarios
5. **Monitoring**: Log all API calls for performance tracking

### Scaling Recommendations

- **Containerization**: Deploy using Docker for consistent environments
- **Load Balancing**: Use multiple API instances behind a load balancer
- **Model Updates**: Implement model versioning and A/B testing workflows
- **Monitoring**: Add application performance monitoring (APM) tools
- **Caching**: Cache model predictions for frequently requested timestamps

## Troubleshooting

### Common Issues

1. **Model Not Loading**
   - Ensure model training completed successfully
   - Check MODEL_SAVE_PATH configuration
   - Verify all model artifacts exist

2. **Feature Engineering Errors**  
   - Validate timestamp format in input data
   - Check for missing columns in raw data
   - Ensure sufficient data for rolling windows

3. **API Validation Errors**
   - Verify timestamp format: YYYY-MM-DDTHH:MM:SSZ
   - Ensure sensor_id matches: TEMP-01
   - Check value is numeric and not NaN

4. **MLflow Connection Issues**
   - Start MLflow UI: mlflow ui
   - Check default port 5000 is available
   - Verify MLflow logging directory permissions

## Future Enhancements

### Planned Features

1. **Multi-Sensor Support**: Extend API to handle multiple sensor types
2. **Advanced Models**: Implement LSTM/Transformer models for complex temporal patterns
3. **Real-time Retraining**: Automated model updates with new data
4. **Alert System**: Integration with notification systems for critical anomalies
5. **Dashboard**: Web interface for monitoring and visualization
6. **Data Drift Detection**: Monitor for changes in data patterns over time

### Research Directions

- **Ensemble Methods**: Combine multiple anomaly detection approaches
- **Explainable AI**: Add interpretability features for anomaly explanations
- **Adaptive Thresholds**: Dynamic threshold adjustment based on historical context
- **Contextual Anomalies**: Incorporate operational context and business rules

## Project Status

**Status**: Production Ready  
**Last Updated**: August 2025  
**Version**: 1.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please contact the development team or open an issue in the repository.

