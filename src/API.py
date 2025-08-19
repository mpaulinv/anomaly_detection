# FastAPI for Anomaly Detection
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import uvicorn
import os

# Import config
from config import MODEL_SAVE_PATH

# Initialize FastAPI app
app = FastAPI(
    title="Anomaly Detection API",
    description="Simple API for detecting anomalies in sensor data using trained Random Forest model",
    version="1.0.0"
)

# Load the trained model at startup
def load_model():
    """Load the trained Random Forest model and metadata"""
    try:
        # Use model path from config (MODEL_SAVE_PATH already includes "random_forest")
        model_dir = MODEL_SAVE_PATH
        print(f"Looking for model in directory: {os.path.abspath(model_dir)}")
        
        # Load model
        model_path = os.path.join(model_dir, "random_forest_model.pkl")
        print(f"Trying to load model from: {os.path.abspath(model_path)}")
        
        if not os.path.exists(model_path):
            print(f"Model file does not exist at: {os.path.abspath(model_path)}")
            print(f"Current working directory: {os.getcwd()}")
            return None, None, None
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "random_forest_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load features
        features_path = os.path.join(model_dir, "random_forest_features.json")
        with open(features_path, 'r') as f:
            features = json.load(f)
        
        print("Model loaded successfully")
        return model, metadata, features['features']
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None, None

# Load model at startup
model, metadata, feature_cols = load_model()

if model is None:
    print("Warning: Model not loaded. Make sure to train the model first!")

# Request/Response models
class SensorData(BaseModel):
    timestamp: str
    sensor_id: str
    value: float
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp format"""
        try:
            # Try to parse the timestamp
            parsed_time = pd.to_datetime(v)
            return v
        except Exception:
            raise ValueError('Invalid timestamp format. Expected format: 2025-08-22T00:00:00Z')
    
    @validator('sensor_id')
    def validate_sensor_id(cls, v):
        """Validate sensor ID"""
        if v != "TEMP-01":
            raise ValueError('Invalid sensor_id. Expected: TEMP-01')
        return v
    
    @validator('value')
    def validate_value(cls, v):
        """Validate value is a number"""
        if not isinstance(v, (int, float)) or np.isnan(v):
            raise ValueError('Value must be a valid number')
        return float(v)

class AnomalyResponse(BaseModel):
    timestamp: str
    sensor_id: str
    value: float
    predicted_value: float
    is_anomaly: bool
    anomaly_score: float
    message: str

def extract_features(timestamp_str: str) -> Dict[str, Any]:
    """Extract features from timestamp"""
    try:
        # Parse timestamp
        dt = pd.to_datetime(timestamp_str)
        
        # Extract features
        features = {
            'hour': dt.hour,
            'minute': dt.minute
        }
        
        return features
        
    except Exception as e:
        raise ValueError(f"Failed to extract features from timestamp: {e}")

def detect_anomaly(features: Dict[str, Any], actual_value: float) -> Dict[str, Any]:
    """Detect if the observation is an anomaly using the trained model"""
    try:
        if model is None:
            raise ValueError("Model not loaded")
        
        # Prepare feature vector for prediction
        # Use only the features that the model was trained on
        X = []
        for feature_name in feature_cols:
            if feature_name in features:
                X.append(features[feature_name])
            else:
                raise ValueError(f"Required feature '{feature_name}' not found")
        
        X = np.array(X).reshape(1, -1)
        
        # Make prediction
        predicted_value = model.predict(X)[0]
        
        # Calculate residual and anomaly score
        residual = abs(actual_value - predicted_value)
        
        # Get anomaly thresholds from metadata
        anomaly_threshold_std = metadata.get('model_params', {}).get('anomaly_threshold_std', 2.5)
        residual_std = metadata.get('model_params', {}).get('residual_std', None)
        
        if residual_std is not None:
            # Use statistical threshold based on training residuals
            anomaly_threshold = residual_std * anomaly_threshold_std
            is_anomaly = residual > anomaly_threshold
            
            # Calculate percentage error for additional info
            percentage_error = (residual / predicted_value) * 100 if predicted_value != 0 else float('inf')
        else:
            # Fallback to percentage-based threshold if residual_std not available
            percentage_error = (residual / predicted_value) * 100 if predicted_value != 0 else float('inf')
            is_anomaly = percentage_error > 10.0
        
        result = {
            'predicted_value': float(predicted_value),
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(residual),
            'percentage_error': float(percentage_error)
        }
        
        return result
        
    except Exception as e:
        raise ValueError(f"Anomaly detection failed: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Anomaly Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": metadata.get('model_type', 'Unknown'),
        "features": feature_cols,
        "training_timestamp": metadata.get('training_timestamp', 'Unknown'),
        "model_params": metadata.get('model_params', {}),
        "training_stats": metadata.get('training_stats', {})
    }

@app.post("/predict", response_model=AnomalyResponse)
async def predict_anomaly(data: SensorData):
    """
    Predict if sensor data is an anomaly
    
    Expected input format:
    {
        "timestamp": "2025-08-22T00:00:00Z",
        "sensor_id": "TEMP-01", 
        "value": 65.32
    }
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
        
        # Extract features from timestamp
        features = extract_features(data.timestamp)
        
        # Detect anomaly
        result = detect_anomaly(features, data.value)
        
        # Prepare response
        message = "Normal observation" if not result['is_anomaly'] else "Anomaly detected!"
        
        response = AnomalyResponse(
            timestamp=data.timestamp,
            sensor_id=data.sensor_id,
            value=data.value,
            predicted_value=result['predicted_value'],
            is_anomaly=result['is_anomaly'],
            anomaly_score=result['anomaly_score'],
            message=message
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/predict/batch")
async def predict_batch(data_list: list[SensorData]):
    """
    Predict anomalies for multiple sensor data points
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        for data in data_list:
            # Extract features
            features = extract_features(data.timestamp)
            
            # Detect anomaly
            result = detect_anomaly(features, data.value)
            
            # Prepare response
            message = "Normal observation" if not result['is_anomaly'] else "Anomaly detected!"
            
            response = AnomalyResponse(
                timestamp=data.timestamp,
                sensor_id=data.sensor_id,
                value=data.value,
                predicted_value=result['predicted_value'],
                is_anomaly=result['is_anomaly'],
                anomaly_score=result['anomaly_score'],
                message=message
            )
            results.append(response)
        
        return {"predictions": results, "total_count": len(results)}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    
    print("Starting Anomaly Detection API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/")
    uvicorn.run(app, host="0.0.0.0", port=8000)