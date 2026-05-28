# polish-energy-forecast-mlops-project
End-to-end MLOps project forecasting Polish energy grid consumption. Includes data preprocessing, feature engineering (sin/cos encoding, lag features), XGBoost training, FastAPI deployment, Docker, and performance monitoring. 156 MW RMSE.


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
