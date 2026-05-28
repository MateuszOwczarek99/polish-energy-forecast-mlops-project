"""
simple monitoring for model performance in production
we log predictions and compare with actual values when available
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """
    track predictions and calculate error metrics
    """
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.predictions_file = self.log_dir / "predictions.csv"
        self.metrics_file = self.log_dir / "metrics.json"
        
        # initialize file if it doesn't exist
        if not self.predictions_file.exists():
            pd.DataFrame(columns=[
                'timestamp', 'prediction', 'actual', 'error', 'abs_error',
                'prediction_time'
            ]).to_csv(self.predictions_file, index=False)
    
    def log_prediction(self, timestamp, prediction):
        """
        record a prediction we made
        """
        
        df = pd.read_csv(self.predictions_file)
        
        new_row = pd.DataFrame({
            'timestamp': [timestamp],
            'prediction': [prediction],
            'actual': [None],
            'error': [None],
            'abs_error': [None],
            'prediction_time': [datetime.now().isoformat()]
        })
        
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.predictions_file, index=False)
        
        logger.info(f"logged prediction for {timestamp}: {prediction:.2f}")
    
    def update_actual(self, timestamp, actual_value):
        """
        when real consumption data arrives, update the record
        """
        
        df = pd.read_csv(self.predictions_file)
        
        mask = df['timestamp'] == timestamp
        if mask.any():
            df.loc[mask, 'actual'] = actual_value
            df.loc[mask, 'error'] = df.loc[mask, 'prediction'] - actual_value
            df.loc[mask, 'abs_error'] = np.abs(df.loc[mask, 'prediction'] - actual_value)
            df.to_csv(self.predictions_file, index=False)
            logger.info(f"updated actual for {timestamp}: {actual_value:.2f}")
        else:
            logger.warning(f"no prediction found for {timestamp}")
    
    def calculate_recent_metrics(self, lookback_days=7):
        """
        calculate error metrics for recent predictions
        """
        
        df = pd.read_csv(self.predictions_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # filter to recent data with actual values
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = df[(df['timestamp'] > cutoff) & (df['actual'].notna())]
        
        if len(recent) == 0:
            return None
        
        metrics = {
            'period_days': lookback_days,
            'n_predictions': len(recent),
            'rmse': np.sqrt((recent['error'] ** 2).mean()),
            'mae': recent['abs_error'].mean(),
            'mape': (recent['abs_error'] / recent['actual']).mean() * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        # save to history
        try:
            with open(self.metrics_file, 'r') as f:
                history = json.load(f)
        except:
            history = []
        
        history.append(metrics)
        
        # keep last 100
        if len(history) > 100:
            history = history[-100:]
        
        with open(self.metrics_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        return metrics
    
    def needs_retraining(self, threshold_mae=150):
        """
        check if model performance has degraded
        """
        
        metrics = self.calculate_recent_metrics(lookback_days=3)
        
        if metrics is None:
            return False
        
        if metrics['mae'] > threshold_mae:
            logger.warning(f"high error detected: mae={metrics['mae']:.2f}")
            return True
        
        return False
    
    def generate_report(self):
        """
        create human readable summary
        """
        
        df = pd.read_csv(self.predictions_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        total = len(df)
        with_actual = df['actual'].notna().sum()
        metrics = self.calculate_recent_metrics()
        
        report = f"""
model monitoring report
generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

summary:
- total predictions: {total}
- with actual values: {with_actual}
- coverage: {with_actual/total*100:.1f}%

recent performance (last 7 days):
"""
        
        if metrics:
            report += f"""
- predictions: {metrics['n_predictions']}
- rmse: {metrics['rmse']:.2f} mw
- mae: {metrics['mae']:.2f} mw
- mape: {metrics['mape']:.1f}%
"""
        else:
            report += "\ninsufficient data for metrics\n"
        
        report += f"\nretraining recommended: {self.needs_retraining()}"
        
        return report

if __name__ == "__main__":
    monitor = ModelMonitor()
    print(monitor.generate_report())
