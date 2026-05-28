"""
data preprocessing - handling missing values and outliers
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    handle all preprocessing steps in a consistent way
    """
    
    def __init__(self, config):
        self.config = config
        self.fill_strategy = config['preprocessing']['fill_strategy']
        self.outlier_strategy = config['preprocessing']['outlier_strategy']
        self.iqr_multiplier = config['preprocessing']['iqr_multiplier']
        self.scaler = None
    
    def handle_missing_values(self, df):
        """
        fill missing values using chosen strategy
        for time series, forward fill usually works well
        """
        
        # count missing before
        missing_before = df.isnull().sum().sum()
        
        if missing_before == 0:
            print("no missing values to handle")
            return df
        
        print(f"handling {missing_before} missing values")
        
        if self.fill_strategy == 'forward':
            # forward fill propagates last known value forward
            df = df.fillna(method='ffill')
            # any remaining at the start, fill backward
            df = df.fillna(method='bfill')
            
        elif self.fill_strategy == 'mean':
            # fill numerical columns with their mean
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].mean())
                
        elif self.fill_strategy == 'median':
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].median())
        
        missing_after = df.isnull().sum().sum()
        print(f"missing values remaining: {missing_after}")
        
        return df
    
    def detect_outliers_iqr(self, df, column):
        """
        find outliers using interquartile range method
        returns boolean mask where true means outlier
        """
        
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        return (df[column] < lower_bound) | (df[column] > upper_bound), lower_bound, upper_bound
    
    def handle_outliers(self, df):
        """
        cap outlier values instead of removing them
        this keeps the data points but limits extreme values
        """
        
        if self.outlier_strategy == 'none':
            print("skipping outlier treatment")
            return df
        
        print(f"handling outliers using {self.outlier_strategy} method")
        
        # only look at numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col == 'consumption':  # focus on target variable
                outliers, lower, upper = self.detect_outliers_iqr(df, col)
                n_outliers = outliers.sum()
                
                if n_outliers > 0 and self.outlier_strategy == 'cap':
                    # cap values at boundaries
                    df.loc[df[col] < lower, col] = lower
                    df.loc[df[col] > upper, col] = upper
                    print(f"  capped {n_outliers} outliers in {col}")
        
        return df
    
    def scale_features(self, df, fit=True):
        """
        standardize features to have mean 0 and standard deviation 1
        this helps models converge faster
        """
        
        # don't scale timestamp or target
        exclude = ['timestamp', self.config['features']['target']]
        columns_to_scale = [c for c in df.columns if c not in exclude]
        
        if fit:
            self.scaler = StandardScaler()
            df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
            print(f"fitted scaler on {len(columns_to_scale)} features")
        else:
            df[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
            print("applied existing scaler")
        
        return df
    
    def run_all(self, df):
        """
        execute complete preprocessing pipeline
        """
        
        print("\nstarting preprocessing...")
        
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)
        # scaling will be done after train/test split
        
        print("preprocessing finished\n")
        
        return df
