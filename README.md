# polish-energy-forecast-mlops-project
End-to-end MLOps project forecasting Polish energy grid consumption. Includes data preprocessing, feature engineering (sin/cos encoding, lag features), XGBoost training, FastAPI deployment, Docker, and performance monitoring. 156 MW RMSE.


# Polish Energy Forecast - MLOps Project

predicting hourly energy consumption for polish power grid using historical data.

## what this project does

we build a machine learning model that forecasts energy demand 24 hours ahead. this helps grid operators plan power generation and avoid shortages.

## how it works

the project follows standard ml pipeline:

1. load hourly consumption data (we generate realistic sample or you can use real data)
2. clean data - handle missing values, remove outliers
3. create features - time features (hour, day of week), lag features (past consumption)
4. train xgboost model
5. serve predictions via rest api
6. monitor model performance over time

## quick start

install dependencies:
```bash
pip install -r requirements.txt





#Project Structure 

```text
polish-energy-mlops/
├── .gitignore
├── README.md
├── requirements.txt
├── Makefile
├── docker-compose.yml 
├── Dockerfile
├── config.yaml
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train_model.py
│   ├── predict.py
│   └── monitor.py
├── notebooks/
│   ├── 01_explore_data.py
│   └── 02_test_model.py
├── tests/
│   └── test_preprocess.py
├── api/
│   └── app.py
├── scripts/
│   └── deploy.sh
└── data/
    ├── raw/
    ├── processed/
    └── external/
