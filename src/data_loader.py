"""
functions for loading and generating energy consumption data
"""

import pandas as pd
import numpy as np
from datetime import datetime

def generate_data(start_date='2020-01-01', end_date='2023-12-31', save_path=None):
    """
    generate synthetic energy consumption data that mimics real patterns
    we use this for demo - in production you would call actual api
    """
    
    # create hourly timestamps
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    n = len(dates)
    
    # set seed for reproducible results
    np.random.seed(42)
    
    # base consumption (typical for poland around 5000 mw)
    base = 5000
    
    # seasonal pattern - higher in winter, lower in summer
    # using sin wave with peak in december/january
    seasonal = 400 * np.sin(2 * np.pi * dates.month / 12 - np.pi/2)
    
    # daily pattern - morning and evening peaks
    # we use sin of hour to create smooth daily curve
    hour_sin = np.sin(2 * np.pi * dates.hour / 24)
    daily = 300 * hour_sin
    
    # weekend effect - lower consumption on saturday and sunday
    is_weekend = (dates.dayofweek >= 5).astype(int)
    weekend_effect = -200 * is_weekend
    
    # add random noise to simulate real world variations
    noise = np.random.normal(0, 80, n)
    
    # combine all components
    consumption = base + seasonal + daily + weekend_effect + noise
    
    # ensure values are realistic (not negative)
    consumption = np.maximum(consumption, 2000)
    
    # create dataframe with additional weather-like features
    df = pd.DataFrame({
        'timestamp': dates,
        'consumption': consumption.round(2),
        'temperature': 10 + 8 * np.sin(2 * np.pi * dates.month / 12) + np.random.normal(0, 3, n),
        'humidity': 70 + 15 * np.sin(2 * np.pi * dates.month / 12) + np.random.normal(0, 10, n),
    })
    
    # clip to realistic ranges
    df['temperature'] = df['temperature'].clip(-15, 35).round(1)
    df['humidity'] = df['humidity'].clip(20, 95).round(1)
    
    # save if path provided
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"data saved to {save_path}")
    
    return df

def load_data(file_path):
    """
    load energy data from csv file
    """
    
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"loaded {len(df)} rows")
    print(f"date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"columns: {list(df.columns)}")
    
    return df

def sanity_check(df):
    """
    perform basic checks on data quality
    """
    
    print("\n--- data sanity check ---")
    
    # check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"warning: found {missing.sum()} missing values")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} missing")
    else:
        print("no missing values found")
    
    # check for duplicate timestamps
    duplicates = df['timestamp'].duplicated().sum()
    if duplicates > 0:
        print(f"warning: found {duplicates} duplicate timestamps")
    else:
        print("no duplicate timestamps")
    
    # check if timestamps are in order
    if df['timestamp'].is_monotonic_increasing:
        print("timestamps are correctly sorted")
    else:
        print("warning: timestamps not sorted - sorting now")
        df = df.sort_values('timestamp')
    
    # check for negative consumption
    if (df['consumption'] < 0).any():
        print("error: found negative consumption values")
    else:
        print("all consumption values positive")
    
    # show basic statistics
    print("\nbasic statistics:")
    print(df['consumption'].describe())
    print("------------------------\n")
    
    return df
