"""
model training using xgboost
we keep it simple but include proper evaluation
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import mlflow
import json
from pathlib import Path

class ModelTrainer:
    """
    train and evaluate xgboost model for regression
    """
    
    def __init__(self, config):
        self.config = config
        self.target = config['features']['target']
        self.model_params = config['model']['parameters']
        self.model = None
    
    def prepare_data(self, df, test_ratio=None):
        """
        split data into train and test sets
        for time series, we use chronological split (no random shuffle)
        """
        
        if test_ratio is None:
            test_ratio = self.config['data']['test_ratio']
        
        # separate features and target
        X = df.drop(columns=[self.target])
        y = df[self.target]
        
        # use last test_ratio% of data as test set
        split_idx = int(len(df) * (1 - test_ratio))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"test set: {X_test.shape[0]} samples")
        
        # save feature names for later use
        feature_names = X_train.columns.tolist()
        with open('models/feature_names.json', 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        train xgboost model with specified parameters
        """
        
        print("\ntraining xgboost model...")
        print(f"parameters: {self.model_params}")
        
        self.model = xgb.XGBRegressor(**self.model_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        
        print("training completed")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        calculate prediction error metrics
        """
        
        print("\nevaluating model...")
        
        y_pred = self.model.predict(X_test)
        
        # calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        # print results
        print(f"rmse: {rmse:.2f} mw")
        print(f"mae: {mae:.2f} mw")
        print(f"r2 score: {r2:.4f}")
        print(f"mape: {mape:.2f}%")
        
        # feature importance
        importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\ntop 10 most important features:")
        for i, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return metrics, y_pred
    
    def save_model(self, path='models/xgboost_model.json'):
        """
        save model to file for later use in api
        """
        
        # create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # save model
        self.model.save_model(path)
        print(f"model saved to {path}")
        
        # also save feature names
        return path
