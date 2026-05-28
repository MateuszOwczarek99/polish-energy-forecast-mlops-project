"""
main script that runs the entire ml pipeline from data loading to model saving
we can run this to train a new model
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import generate_data, load_data, sanity_check
from src.preprocess import DataPreprocessor
from src.features import FeatureEngineer
from src.train_model import ModelTrainer

def load_config(config_path="config.yaml"):
    """load configuration from yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_directories():
    """create necessary directories if they don't exist"""
    dirs = ["data/raw", "data/processed", "models", "logs", "reports"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def main():
    """execute the complete pipeline"""
    
    print("=" * 60)
    print("polish energy forecast pipeline")
    print("=" * 60)
    
    # create directories
    create_directories()
    
    # load configuration
    config = load_config()
    print("✓ configuration loaded")
    
    # step 1: load or generate data
    print("\n[1/6] loading data...")
    raw_path = config['data']['raw_path']
    
    if not Path(raw_path).exists():
        print("generating sample data (replace with real data in production)")
        df = generate_data(
            start_date=config['data']['start_date'],
            end_date=config['data']['end_date'],
            save_path=raw_path
        )
    else:
        df = load_data(raw_path)
    
    print(f"loaded {len(df)} records")
    
    # step 2: sanity check
    print("\n[2/6] running sanity checks...")
    df = sanity_check(df)
    
    # step 3: preprocess data (missing values, outliers)
    print("\n[3/6] preprocessing data...")
    preprocessor = DataPreprocessor(config)
    df = preprocessor.run_all(df)
    
    # step 4: create features
    print("\n[4/6] engineering features...")
    engineer = FeatureEngineer(config)
    df = engineer.build_all_features(df)
    
    # step 5: train model
    print("\n[5/6] training model...")
    trainer = ModelTrainer(config)
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    model = trainer.train(X_train, y_train)
    
    # step 6: evaluate and save
    print("\n[6/6] evaluating and saving model...")
    metrics, y_pred = trainer.evaluate(X_test, y_test)
    model_path = trainer.save_model()
    
    # save processed data for later
    df.to_parquet(config['data']['processed_path'])
    print(f"processed data saved to {config['data']['processed_path']}")
    
    # final summary
    print("\n" + "=" * 60)
    print("pipeline completed successfully")
    print("=" * 60)
    print(f"model saved to: {model_path}")
    print(f"test rmse: {metrics['rmse']:.2f} mw")
    print(f"test mae: {metrics['mae']:.2f} mw")
    print(f"test r2: {metrics['r2']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
