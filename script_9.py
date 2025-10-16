# Create model trainer module
model_trainer_content = '''"""
Model training orchestrator for system performance optimization
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import mlflow
import mlflow.sklearn
import mlflow.keras
from pathlib import Path
import joblib

from .time_series_forecaster import LSTMForecaster
from .anomaly_detector import SystemAnomalyDetector
from ..data_collection.system_monitor import SystemMonitor
from ..data_collection.data_preprocessor import SystemDataPreprocessor
from ..config.settings import get_config
from ..config.logging_config import model_logger

class ModelTrainer:
    """
    Comprehensive model training system for system performance optimization
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.system_monitor = SystemMonitor(config)
        self.preprocessor = SystemDataPreprocessor(config)
        self.lstm_forecaster = LSTMForecaster(config)
        self.anomaly_detector = SystemAnomalyDetector(config)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        self._setup_mlflow_experiment()
    
    def _setup_mlflow_experiment(self):
        """Setup MLflow experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(self.config.MLFLOW_EXPERIMENT_NAME)
            if experiment is None:
                mlflow.create_experiment(self.config.MLFLOW_EXPERIMENT_NAME)
            mlflow.set_experiment(self.config.MLFLOW_EXPERIMENT_NAME)
            model_logger.logger.info(f"MLflow experiment set: {self.config.MLFLOW_EXPERIMENT_NAME}")
        except Exception as e:
            model_logger.logger.error(f"Error setting up MLflow experiment: {e}")
    
    def collect_training_data(self, duration_hours: int = 24) -> pd.DataFrame:
        """
        Collect training data from system monitoring
        
        Args:
            duration_hours: Hours of data to collect
            
        Returns:
            DataFrame with system metrics
        """
        model_logger.logger.info(f"Collecting {duration_hours} hours of training data")
        
        # Get historical data
        df = self.system_monitor.get_historical_data(hours=duration_hours)
        
        if df.empty:
            model_logger.logger.warning("No historical data available, starting fresh collection")
            # Start monitoring to collect fresh data
            self.system_monitor.start_monitoring()
            # For demo purposes, we'll create some synthetic data
            df = self._generate_synthetic_data(duration_hours)
        
        model_logger.logger.info(f"Collected {len(df)} samples for training")
        return df
    
    def _generate_synthetic_data(self, duration_hours: int) -> pd.DataFrame:
        """
        Generate synthetic system data for demonstration
        
        Args:
            duration_hours: Hours of data to generate
            
        Returns:
            DataFrame with synthetic system metrics
        """
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=duration_hours)
        timestamps = pd.date_range(start_time, end_time, freq='30S')
        
        # Generate synthetic metrics with realistic patterns
        np.random.seed(42)
        n_samples = len(timestamps)
        
        # Base patterns with daily/hourly cycles
        hours = np.array([t.hour for t in timestamps])
        daily_pattern = 50 + 30 * np.sin(2 * np.pi * hours / 24)
        
        data = {
            'timestamp': timestamps,
            'cpu_percent': np.clip(daily_pattern + np.random.normal(0, 10, n_samples), 0, 100),
            'memory_percent': np.clip(40 + 20 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, n_samples), 0, 100),
            'disk_usage_percent': np.clip(60 + np.random.normal(0, 2, n_samples), 0, 100),
            'disk_io_read': np.cumsum(np.random.exponential(1000, n_samples)),
            'disk_io_write': np.cumsum(np.random.exponential(800, n_samples)),
            'network_io_sent': np.cumsum(np.random.exponential(500, n_samples)),
            'network_io_recv': np.cumsum(np.random.exponential(600, n_samples)),
            'boot_time': [datetime.now().timestamp() - 86400] * n_samples,  # 1 day ago
            'process_count': np.random.poisson(150, n_samples),
            'load_average_1min': np.clip(np.random.gamma(2, 0.5, n_samples), 0, 10),
            'load_average_5min': np.clip(np.random.gamma(2, 0.4, n_samples), 0, 10),
            'load_average_15min': np.clip(np.random.gamma(2, 0.3, n_samples), 0, 10),
            'memory_available': 8 * 1024**3 - (np.clip(40 + 20 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, n_samples), 0, 100) / 100) * 8 * 1024**3
        }
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        for idx in anomaly_indices:
            data['cpu_percent'][idx] = np.random.uniform(85, 98)
            data['memory_percent'][idx] = np.random.uniform(90, 95)
        
        df = pd.DataFrame(data)
        df['memory_available'] = df['memory_available'].astype(int)
        
        return df
    
    def train_forecasting_model(self, df: pd.DataFrame, target_column: str = 'cpu_percent') -> Dict[str, Any]:
        """
        Train LSTM forecasting model
        
        Args:
            df: Training data
            target_column: Column to forecast
            
        Returns:
            Training results
        """
        with mlflow.start_run(run_name=f"lstm_forecasting_{target_column}"):
            model_logger.logger.info(f"Training LSTM forecaster for {target_column}")
            
            # Log parameters
            mlflow.log_param("model_type", "LSTM")
            mlflow.log_param("target_column", target_column)
            mlflow.log_param("lookback_window", self.config.LSTM_LOOKBACK_WINDOW)
            mlflow.log_param("forecast_horizon", self.config.FORECAST_HORIZON)
            
            # Preprocess data
            df_processed = self.preprocessor.create_features(df)
            
            # Prepare time series data
            X, y = self.preprocessor.prepare_time_series_data(
                df_processed, 
                target_column,
                self.config.LSTM_LOOKBACK_WINDOW,
                self.config.FORECAST_HORIZON
            )
            
            # Split data
            split_data = self.preprocessor.split_data(X, y, time_series_split=True)
            
            # Scale features
            X_train_scaled, X_val_scaled = self.preprocessor.scale_features(
                split_data['X_train'].reshape(-1, split_data['X_train'].shape[-1]),
                split_data['X_val'].reshape(-1, split_data['X_val'].shape[-1])
            )
            
            # Reshape back for LSTM
            X_train_scaled = X_train_scaled.reshape(split_data['X_train'].shape)
            X_val_scaled = X_val_scaled.reshape(split_data['X_val'].shape)
            
            # Build and train model
            self.lstm_forecaster.build_model(
                input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
                output_size=self.config.FORECAST_HORIZON
            )
            
            history = self.lstm_forecaster.train(
                X_train_scaled, split_data['y_train'],
                X_val_scaled, split_data['y_val']
            )
            
            # Evaluate model
            X_test_scaled = self.preprocessor.scalers['standard'].transform(
                split_data['X_test'].reshape(-1, split_data['X_test'].shape[-1])
            ).reshape(split_data['X_test'].shape)
            
            metrics = self.lstm_forecaster.evaluate(X_test_scaled, split_data['y_test'])
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Save model
            model_path = self.config.get_paths()['models'] / f'lstm_{target_column}.h5'
            self.lstm_forecaster.save_model(str(model_path))
            mlflow.keras.log_model(self.lstm_forecaster.model, "lstm_model")
            
            # Save preprocessor
            preprocessor_path = self.config.get_paths()['models'] / f'preprocessor_{target_column}.pkl'
            self.preprocessor.save_preprocessor(str(preprocessor_path))
            
            results = {
                'model_path': str(model_path),
                'preprocessor_path': str(preprocessor_path),
                'metrics': metrics,
                'history': history
            }
            
            model_logger.logger.info(f"LSTM forecaster training completed for {target_column}")
            return results
    
    def train_anomaly_detection_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train anomaly detection model
        
        Args:
            df: Training data (normal system behavior)
            
        Returns:
            Training results
        """
        with mlflow.start_run(run_name="anomaly_detection"):
            model_logger.logger.info("Training anomaly detection model")
            
            # Log parameters
            mlflow.log_param("model_type", "Ensemble_Anomaly_Detection")
            mlflow.log_param("contamination", 0.1)
            
            # Preprocess data
            df_processed = self.preprocessor.create_anomaly_detection_features(df)
            
            # Select numeric features
            numeric_features = df_processed.select_dtypes(include=[np.number]).columns
            numeric_features = [col for col in numeric_features if col not in ['timestamp']]
            
            X = df_processed[numeric_features].values
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            
            # Scale features
            X_scaled, _ = self.preprocessor.scale_features(X)
            
            # Train anomaly detection model
            self.anomaly_detector.fit(X_scaled)
            
            # Test on the training data to get baseline metrics
            anomaly_results = self.anomaly_detector.detect_anomalies(X_scaled)
            
            # Log metrics
            mlflow.log_metric("anomaly_rate", anomaly_results['anomaly_rate'])
            mlflow.log_metric("total_anomalies", float(anomaly_results['total_anomalies']))
            
            # Save model
            model_path = self.config.get_paths()['models'] / 'anomaly_detector.pkl'
            self.anomaly_detector.save_model(str(model_path))
            mlflow.sklearn.log_model(self.anomaly_detector.ensemble, "anomaly_model")
            
            results = {
                'model_path': str(model_path),
                'anomaly_results': anomaly_results,
                'feature_columns': numeric_features.tolist()
            }
            
            model_logger.logger.info("Anomaly detection model training completed")
            return results
    
    def full_training_pipeline(self, duration_hours: int = 24) -> Dict[str, Any]:
        """
        Execute complete training pipeline
        
        Args:
            duration_hours: Hours of data to collect for training
            
        Returns:
            Complete training results
        """
        model_logger.logger.info("Starting full training pipeline")
        
        # Collect training data
        df = self.collect_training_data(duration_hours)
        
        if df.empty:
            raise ValueError("No training data available")
        
        results = {
            'data_info': {
                'samples': len(df),
                'features': len(df.columns),
                'time_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                }
            },
            'models': {}
        }
        
        # Train forecasting models for key metrics
        key_metrics = ['cpu_percent', 'memory_percent']
        for metric in key_metrics:
            if metric in df.columns:
                try:
                    forecast_results = self.train_forecasting_model(df, metric)
                    results['models'][f'forecasting_{metric}'] = forecast_results
                except Exception as e:
                    model_logger.logger.error(f"Error training forecaster for {metric}: {e}")
        
        # Train anomaly detection model
        try:
            anomaly_results = self.train_anomaly_detection_model(df)
            results['models']['anomaly_detection'] = anomaly_results
        except Exception as e:
            model_logger.logger.error(f"Error training anomaly detector: {e}")
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'config': {
                'monitoring_interval': self.config.MONITORING_INTERVAL,
                'lstm_lookback_window': self.config.LSTM_LOOKBACK_WINDOW,
                'forecast_horizon': self.config.FORECAST_HORIZON,
                'anomaly_threshold': self.config.ANOMALY_THRESHOLD
            },
            'results': results
        }
        
        metadata_path = self.config.get_paths()['models'] / 'training_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        model_logger.logger.info("Full training pipeline completed successfully")
        return results
    
    def retrain_if_needed(self) -> bool:
        """
        Check if models need retraining based on performance metrics
        
        Returns:
            True if retraining was performed
        """
        # Check last training time
        metadata_path = self.config.get_paths()['models'] / 'training_metadata.json'
        
        if not metadata_path.exists():
            model_logger.logger.info("No training metadata found, triggering retraining")
            self.full_training_pipeline()
            return True
        
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        last_training = datetime.fromisoformat(metadata['training_date'])
        time_since_training = datetime.now() - last_training
        
        if time_since_training.total_seconds() > self.config.MODEL_RETRAIN_INTERVAL:
            model_logger.logger.info(f"Models last trained {time_since_training} ago, triggering retraining")
            self.full_training_pipeline()
            return True
        
        # TODO: Add performance-based retraining logic
        # - Check model drift
        # - Check prediction accuracy
        # - Check anomaly detection performance
        
        model_logger.logger.info("Models are up to date, no retraining needed")
        return False
    
    def load_trained_models(self) -> Dict[str, Any]:
        """
        Load all trained models
        
        Returns:
            Dictionary of loaded models
        """
        models = {}
        models_path = self.config.get_paths()['models']
        
        try:
            # Load LSTM models
            for metric in ['cpu_percent', 'memory_percent']:
                model_file = models_path / f'lstm_{metric}.h5'
                preprocessor_file = models_path / f'preprocessor_{metric}.pkl'
                
                if model_file.exists() and preprocessor_file.exists():
                    forecaster = LSTMForecaster(self.config)
                    forecaster.load_model(str(model_file))
                    
                    preprocessor = SystemDataPreprocessor(self.config)
                    preprocessor.load_preprocessor(str(preprocessor_file))
                    
                    models[f'forecasting_{metric}'] = {
                        'model': forecaster,
                        'preprocessor': preprocessor
                    }
            
            # Load anomaly detection model
            anomaly_model_file = models_path / 'anomaly_detector.pkl'
            if anomaly_model_file.exists():
                anomaly_detector = SystemAnomalyDetector(self.config)
                anomaly_detector.load_model(str(anomaly_model_file))
                models['anomaly_detection'] = anomaly_detector
            
            model_logger.logger.info(f"Loaded {len(models)} trained models")
            
        except Exception as e:
            model_logger.logger.error(f"Error loading models: {e}")
        
        return models

# Global model trainer instance
model_trainer = ModelTrainer()
'''

with open('ml_system_optimizer/src/models/model_trainer.py', 'w') as f:
    f.write(model_trainer_content)

print("✅ model_trainer.py created successfully!")