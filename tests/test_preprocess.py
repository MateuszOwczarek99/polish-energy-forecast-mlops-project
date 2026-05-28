"""
unit tests for preprocessing functions
"""

import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from src.preprocess import DataPreprocessor

def test_missing_value_handling():
    """
    test that missing values are properly filled
    """
    
    config = {
        'preprocessing': {
            'fill_strategy': 'forward',
            'outlier_strategy': 'cap',
            'iqr_multiplier': 1.5
        },
        'features': {'target': 'consumption'}
    }
    
    preprocessor = DataPreprocessor(config)
    
    # create test data with missing values
    df = pd.DataFrame({
        'consumption': [100, np.nan, 200, np.nan, 300],
        'temperature': [20, 21, np.nan, 22, 23]
    })
    
    df_clean = preprocessor.handle_missing_values(df)
    
    # should have no missing values
    assert df_clean.isnull().sum().sum() == 0

def test_outlier_capping():
    """
    test that outliers are capped within bounds
    """
    
    config = {
        'preprocessing': {
            'fill_strategy': 'forward',
            'outlier_strategy': 'cap',
            'iqr_multiplier': 1.5
        },
        'features': {'target': 'consumption'}
    }
    
    preprocessor = DataPreprocessor(config)
    
    # create test data with obvious outlier
    df = pd.DataFrame({
        'consumption': [100, 105, 110, 500, 108, 102, 95],
        'temperature': [20, 21, 22, 23, 21, 20, 22]
    })
    
    df_capped = preprocessor.handle_outliers(df)
    
    # outlier should be capped
    assert df_capped['consumption'].max() < 200

if __name__ == "__main__":
    pytest.main()
