"""
fastapi application for serving predictions
we load the trained model and provide http endpoints
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import xgboost as xgb
import logging

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# request and response models
class PredictionRequest(BaseModel):
    timestamp: str  # format: "2024-01-15 14:00:00"
    temperature: Optional[float] = None
    humidity: Optional[float] = None

class PredictionResponse(BaseModel):
    timestamp: str
    predicted_consumption_mw: float
    model_version: str

class BatchRequest(BaseModel):
    requests: List[PredictionRequest]

# global variables
model = None
feature_names = None
model_version = "1.0.0"

def create_features(timestamp_str, temperature=None, humidity=None):
    """
    recreate the same features we used during training
    this must match exactly what feature_engineering.py does
    """
    
    dt = pd.to_datetime(timestamp_str)
    
    # basic time features
    hour = dt.hour
    dayofweek = dt.dayofweek
    month = dt.month
    is_weekend = 1 if dayofweek >= 5 else 0
    
    # cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dayofweek / 7)
    dow_cos = np.cos(2 * np.pi * dayofweek / 7)
    
    # season
    if month in [12, 1, 2]:
        season = 0
    elif month in [3, 4, 5]:
        season = 1
    elif month in [6, 7, 8]:
        season = 2
    else:
        season = 3
    
    # create dataframe with one row
    features = pd.DataFrame({
        'hour': [hour],
        'dayofweek': [dayofweek],
        'month': [month],
        'is_weekend': [is_weekend],
        'hour_sin': [hour_sin],
        'hour_cos': [hour_cos],
        'dow_sin': [dow_sin],
        'dow_cos': [dow_cos],
        'season': [season],
        'temperature': [temperature if temperature is not None else 10.0],
        'humidity': [humidity if humidity is not None else 65.0],
    })
    
    # add lag features with default values
    # in a real system, we would query last known values from database
    features['lag_1h'] = 5000.0
    features['lag_2h'] = 5000.0
    features['lag_3h'] = 5000.0
    features['lag_6h'] = 5000.0
    features['lag_12h'] = 5000.0
    features['lag_24h'] = 5000.0
    features['rolling_mean_6h'] = 5000.0
    features['rolling_mean_12h'] = 5000.0
    features['rolling_mean_24h'] = 5000.0
    features['rolling_std_6h'] = 100.0
    features['rolling_std_12h'] = 100.0
    features['rolling_std_24h'] = 100.0
    features['diff_24h'] = 0.0
    features['pct_change_24h'] = 0.0
    features['diff_1h'] = 0.0
    
    # ensure columns are in same order as training
    if feature_names:
        features = features[feature_names]
    
    return features

# create fastapi app
app = FastAPI(
    title="Energy Forecast API",
    description="predict polish energy consumption for any hour",
    version="1.0.0"
)

@app.on_event("startup")
async def load_model():
    """
    load model when api starts
    """
    global model, feature_names
    
    model_path = os.getenv('MODEL_PATH', 'models/xgboost_model.json')
    
    if os.path.exists(model_path):
        try:
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            logger.info(f"model loaded from {model_path}")
            
            # load feature names if available
            if os.path.exists('models/feature_names.json'):
                with open('models/feature_names.json', 'r') as f:
                    feature_names = json.load(f)
                logger.info(f"loaded {len(feature_names)} feature names")
        except Exception as e:
            logger.error(f"failed to load model: {e}")
            model = None
    else:
        logger.warning(f"model not found at {model_path}")
        model = None

@app.get("/health")
async def health_check():
    """
    simple health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    predict energy consumption for a single timestamp
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    
    try:
        # create features
        features = create_features(
            request.timestamp,
            request.temperature,
            request.humidity
        )
        
        # make prediction
        prediction = model.predict(features)[0]
        
        logger.info(f"prediction for {request.timestamp}: {prediction:.2f} mw")
        
        return PredictionResponse(
            timestamp=request.timestamp,
            predicted_consumption_mw=round(prediction, 2),
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchRequest):
    """
    predict for multiple timestamps
    """
    
    responses = []
    for req in request.requests:
        try:
            response = await predict(req)
            responses.append(response)
        except Exception as e:
            logger.error(f"failed for {req.timestamp}: {e}")
            # continue with other predictions
    
    return responses

@app.get("/info")
async def model_info():
    """
    get information about loaded model
    """
    
    return {
        "model_loaded": model is not None,
        "model_version": model_version,
        "features_expected": len(feature_names) if feature_names else 0
    }
