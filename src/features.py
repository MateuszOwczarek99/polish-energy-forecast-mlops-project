"""
feature engineering for time series forecasting
we create features from timestamps and past values
"""

import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    build features that help the model learn patterns
    """
    
    def __init__(self, config):
        self.config = config
        self.target = config['features']['target']
        self.lag_hours = config['features']['lag_hours']
        self.rolling_windows = config['features']['rolling_windows']
    
    def create_time_features(self, df):
        """
        extract useful information from timestamp
        we use sin/cos for hour to preserve cyclical nature
        """
        
        print("creating time-based features...")
        
        # basic time components
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # weekend flag (binary feature)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # cyclical encoding for hour
        # we use sin and cos so that hour 23 and 0 are close
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # season from month
        def get_season(month):
            if month in [12, 1, 2]:
                return 0  # winter
            elif month in [3, 4, 5]:
                return 1  # spring
            elif month in [6, 7, 8]:
                return 2  # summer
            else:
                return 3  # fall
        
        df['season'] = df['month'].apply(get_season)
        
        return df
    
    def create_lag_features(self, df):
        """
        use past consumption values as features
        this is usually the most important set of features
        """
        
        print(f"creating lag features for hours: {self.lag_hours}")
        
        for lag in self.lag_hours:
            df[f'lag_{lag}h'] = df[self.target].shift(lag)
        
        return df
    
    def create_rolling_features(self, df):
        """
        create moving averages and standard deviations
        these capture recent trends
        """
        
        print(f"creating rolling features for windows: {self.rolling_windows}")
        
        for window in self.rolling_windows:
            # mean over window
            df[f'rolling_mean_{window}h'] = df[self.target].rolling(window=window).mean()
            # standard deviation over window (volatility)
            df[f'rolling_std_{window}h'] = df[self.target].rolling(window=window).std()
        
        return df
    
    def create_rate_features(self, df):
        """
        calculate how consumption changes over time
        """
        
        # difference from 24 hours ago (same hour yesterday)
        df['diff_24h'] = df[self.target] - df[self.target].shift(24)
        
        # percentage change
        df['pct_change_24h'] = (df[self.target] - df[self.target].shift(24)) / df[self.target].shift(24) * 100
        
        # difference from previous hour
        df['diff_1h'] = df[self.target].diff(1)
        
        return df
    
    def remove_nan_rows(self, df):
        """
        lag and rolling features create nan values at the start
        we need to remove these rows
        """
        
        before = len(df)
        df = df.dropna()
        after = len(df)
        
        removed = before - after
        if removed > 0:
            print(f"removed {removed} rows with nan values")
        
        return df
    
    def build_all_features(self, df):
        """
        run complete feature engineering pipeline
        """
        
        print("\nbuilding features...")
        
        # work on a copy
        df = df.copy()
        
        # create features in order
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_rate_features(df)
        
        # clean up
        df = self.remove_nan_rows(df)
        
        # drop original timestamp (model doesn't need it)
        if 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        
        print(f"final feature set: {df.shape[1]} columns, {df.shape[0]} rows")
        print("feature engineering done\n")
        
        return df
