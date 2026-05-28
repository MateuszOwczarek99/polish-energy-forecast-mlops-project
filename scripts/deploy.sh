#!/bin/bash
# simple deployment script

echo "deploying energy forecast model..."

# train model
echo "training model..."
python run_pipeline.py

# check if model was created
if [ -f "models/xgboost_model.json" ]; then
    echo "model training successful"
else
    echo "model training failed"
    exit 1
fi

# build docker image
echo "building docker image..."
docker build -t energy-forecast-api .

# run container
echo "starting container..."
docker run -d -p 8000:8000 --name energy-api energy-forecast-api

echo "deployment complete"
echo "api available at http://localhost:8000"
echo "check health at http://localhost:8000/health"
