# Create data preprocessing module
data_preprocessor_content = '''"""
Data preprocessing module for system performance data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional, Any
import joblib
from pathlib import Path

from ..config.settings import get_config
from ..config.logging_config import get_logger

logger = get_logger('data_preprocessor')

class SystemDataPreprocessor:
    """
    Comprehensive data preprocessing for system performance metrics
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = []
        
    def prepare_time_series_data(self, df: pd.DataFrame, 
                                target_column: str,
                                lookback_window: int = None,
                                forecast_horizon: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data for LSTM training
        
        Args:
            df: DataFrame with time series data
            target_column: Column to predict
            lookback_window: Number of time steps to look back
            forecast_horizon: Number of steps to forecast ahead
            
        Returns:
            Tuple of (X, y) arrays ready for training
        """
        lookback_window = lookback_window or self.config.LSTM_LOOKBACK_WINDOW
        forecast_horizon = forecast_horizon or self.config.FORECAST_HORIZON
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Select relevant features
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'id']]
        data = df[feature_cols].values
        target_data = df[target_column].values
        
        X, y = [], []
        
        for i in range(lookback_window, len(data) - forecast_horizon + 1):
            # Input sequence
            X.append(data[i-lookback_window:i])
            # Target sequence
            y.append(target_data[i:i+forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from raw system metrics
        
        Args:
            df: DataFrame with raw metrics
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Rolling statistics (for trend analysis)
        for window in [5, 10, 30]:
            df[f'cpu_percent_rolling_mean_{window}'] = df['cpu_percent'].rolling(window=window).mean()
            df[f'memory_percent_rolling_mean_{window}'] = df['memory_percent'].rolling(window=window).mean()
            df[f'cpu_percent_rolling_std_{window}'] = df['cpu_percent'].rolling(window=window).std()
            df[f'memory_percent_rolling_std_{window}'] = df['memory_percent'].rolling(window=window).std()
        
        # Rate of change features
        df['cpu_percent_diff'] = df['cpu_percent'].diff()
        df['memory_percent_diff'] = df['memory_percent'].diff()
        df['disk_usage_percent_diff'] = df['disk_usage_percent'].diff()
        
        # Resource utilization ratios
        total_memory = df['memory_available'].max() + (df['memory_percent'].max() / 100) * df['memory_available'].max()
        df['memory_used_gb'] = (df['memory_percent'] / 100) * total_memory / (1024**3)
        
        # Network and disk I/O rates (bytes per second)
        df['disk_io_total'] = df['disk_io_read'] + df['disk_io_write']
        df['network_io_total'] = df['network_io_sent'] + df['network_io_recv']
        
        # I/O rates (calculate differences and divide by time interval)
        time_diff = df['timestamp'].diff().dt.total_seconds()
        df['disk_read_rate'] = df['disk_io_read'].diff() / time_diff
        df['disk_write_rate'] = df['disk_io_write'].diff() / time_diff
        df['network_sent_rate'] = df['network_io_sent'].diff() / time_diff
        df['network_recv_rate'] = df['network_io_recv'].diff() / time_diff
        
        # System load indicators
        df['high_cpu_usage'] = (df['cpu_percent'] > 80).astype(int)
        df['high_memory_usage'] = (df['memory_percent'] > 80).astype(int)
        df['high_disk_usage'] = (df['disk_usage_percent'] > 80).astype(int)
        
        # Composite performance score
        df['performance_score'] = (
            (100 - df['cpu_percent']) * 0.3 +
            (100 - df['memory_percent']) * 0.3 +
            (100 - df['disk_usage_percent']) * 0.2 +
            (df['load_average_1min'] < 1).astype(int) * 20
        )
        
        # Drop NaN values created by rolling operations and differences
        df = df.dropna()
        
        logger.info(f"Created {len(df.columns)} features from {len(df)} samples")
        return df
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray = None, 
                      scaler_type: str = 'standard') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using specified scaler
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple of scaled (X_train, X_test)
        """
        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if scaler_type not in scaler_map:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        scaler = scaler_map[scaler_type]
        
        # Fit scaler on training data
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Store scaler for later use
        self.scalers[scaler_type] = scaler
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Applied {scaler_type} scaling to features")
        return X_train_scaled, X_test_scaled
    
    def create_anomaly_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specifically for anomaly detection
        
        Args:
            df: DataFrame with system metrics
            
        Returns:
            DataFrame with anomaly detection features
        """
        df = df.copy()
        
        # Z-scores for all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['timestamp', 'id']:
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[f'{col}_zscore'] = (df[col] - mean_val) / std_val
        
        # Isolation features
        # Distance from recent mean
        for window in [10, 30, 60]:
            for col in ['cpu_percent', 'memory_percent', 'disk_usage_percent']:
                rolling_mean = df[col].rolling(window=window).mean()
                df[f'{col}_deviation_{window}'] = np.abs(df[col] - rolling_mean)
        
        # Sudden change detection
        for col in ['cpu_percent', 'memory_percent']:
            df[f'{col}_sudden_change'] = np.abs(df[col].diff()) > (df[col].std() * 2)
        
        logger.info(f"Created anomaly detection features for {len(df)} samples")
        return df
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2,
                   validation_size: float = 0.2,
                   time_series_split: bool = True) -> Dict[str, np.ndarray]:
        """
        Split data into training, validation, and test sets
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            time_series_split: Whether to use time-series aware splitting
            
        Returns:
            Dictionary with train/val/test splits
        """
        if time_series_split:
            # For time series, use temporal splitting
            n_samples = len(X)
            n_test = int(n_samples * test_size)
            n_val = int((n_samples - n_test) * validation_size)
            
            X_temp = X[:-n_test]
            y_temp = y[:-n_test]
            X_test = X[-n_test:]
            y_test = y[-n_test:]
            
            X_train = X_temp[:-n_val]
            y_train = y_temp[:-n_val]
            X_val = X_temp[-n_val:]
            y_val = y_temp[-n_val:]
            
        else:
            # Random splitting
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=validation_size, random_state=42
            )
        
        split_data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return split_data
    
    def save_preprocessor(self, filepath: str):
        """
        Save preprocessor state (scalers, feature columns, etc.)
        
        Args:
            filepath: Path to save the preprocessor
        """
        preprocessor_state = {
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }
        
        joblib.dump(preprocessor_state, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """
        Load preprocessor state
        
        Args:
            filepath: Path to load the preprocessor from
        """
        preprocessor_state = joblib.load(filepath)
        self.scalers = preprocessor_state['scalers']
        self.feature_columns = preprocessor_state['feature_columns']
        self.target_columns = preprocessor_state['target_columns']
        
        logger.info(f"Preprocessor loaded from {filepath}")
    
    def prepare_for_prediction(self, df: pd.DataFrame, target_column: str = None) -> np.ndarray:
        """
        Prepare new data for prediction using saved scalers
        
        Args:
            df: DataFrame with new data
            target_column: Target column (for feature creation)
            
        Returns:
            Preprocessed data ready for prediction
        """
        # Create features
        df_processed = self.create_features(df)
        
        # Select relevant feature columns
        if self.feature_columns:
            df_processed = df_processed[self.feature_columns]
        
        # Apply scaling if scalers are available
        if 'standard' in self.scalers:
            scaler = self.scalers['standard']
            X_scaled = scaler.transform(df_processed)
            return X_scaled
        
        return df_processed.values

# Global preprocessor instance
preprocessor = SystemDataPreprocessor()
'''

with open('ml_system_optimizer/src/data_collection/data_preprocessor.py', 'w') as f:
    f.write(data_preprocessor_content)

print("✅ data_preprocessor.py created successfully!")