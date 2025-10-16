# Create comprehensive settings.py configuration
settings_content = '''"""
Configuration settings for ML System Optimizer
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

class Config:
    """Base configuration class"""
    
    # System Monitoring Settings
    MONITORING_INTERVAL = int(os.getenv('MONITORING_INTERVAL', 5))  # seconds
    DATA_COLLECTION_DURATION = int(os.getenv('DATA_COLLECTION_DURATION', 3600))  # 1 hour
    
    # Model Settings
    LSTM_LOOKBACK_WINDOW = int(os.getenv('LSTM_LOOKBACK_WINDOW', 60))  # 60 time steps
    FORECAST_HORIZON = int(os.getenv('FORECAST_HORIZON', 12))  # 12 steps ahead
    MODEL_RETRAIN_INTERVAL = int(os.getenv('MODEL_RETRAIN_INTERVAL', 86400))  # 24 hours
    
    # Anomaly Detection Settings
    ANOMALY_THRESHOLD = float(os.getenv('ANOMALY_THRESHOLD', 2.5))  # Z-score threshold
    ANOMALY_DETECTION_METHODS = ['isolation_forest', 'lstm_autoencoder', 'statistical']
    
    # Database Settings
    DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{BASE_DIR}/system_data.db')
    
    # Flask API Settings
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # MLflow Settings
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', f'file://{BASE_DIR}/mlruns')
    MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'system_performance_optimization')
    
    # System Resource Thresholds
    CPU_WARNING_THRESHOLD = float(os.getenv('CPU_WARNING_THRESHOLD', 80.0))
    CPU_CRITICAL_THRESHOLD = float(os.getenv('CPU_CRITICAL_THRESHOLD', 90.0))
    MEMORY_WARNING_THRESHOLD = float(os.getenv('MEMORY_WARNING_THRESHOLD', 80.0))
    MEMORY_CRITICAL_THRESHOLD = float(os.getenv('MEMORY_CRITICAL_THRESHOLD', 90.0))
    DISK_WARNING_THRESHOLD = float(os.getenv('DISK_WARNING_THRESHOLD', 80.0))
    DISK_CRITICAL_THRESHOLD = float(os.getenv('DISK_CRITICAL_THRESHOLD', 90.0))
    
    # Model Performance Thresholds
    MODEL_ACCURACY_THRESHOLD = float(os.getenv('MODEL_ACCURACY_THRESHOLD', 0.85))
    MODEL_DRIFT_THRESHOLD = float(os.getenv('MODEL_DRIFT_THRESHOLD', 0.1))
    
    # Logging Settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', str(BASE_DIR / 'logs' / 'system_optimizer.log'))
    
    # Monitoring Dashboard Settings
    DASHBOARD_UPDATE_INTERVAL = int(os.getenv('DASHBOARD_UPDATE_INTERVAL', 10))  # seconds
    DASHBOARD_HISTORY_WINDOW = int(os.getenv('DASHBOARD_HISTORY_WINDOW', 3600))  # 1 hour
    
    # Performance Optimization Settings
    AUTO_SCALING_ENABLED = os.getenv('AUTO_SCALING_ENABLED', 'False').lower() == 'true'
    RESOURCE_OPTIMIZATION_ENABLED = os.getenv('RESOURCE_OPTIMIZATION_ENABLED', 'True').lower() == 'true'
    
    # Alert Settings
    ALERT_EMAIL_ENABLED = os.getenv('ALERT_EMAIL_ENABLED', 'False').lower() == 'true'
    ALERT_EMAIL_RECIPIENTS = os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(',')
    ALERT_WEBHOOK_URL = os.getenv('ALERT_WEBHOOK_URL', '')
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            'lstm': {
                'units': [64, 32],
                'dropout': 0.2,
                'epochs': 100,
                'batch_size': 32,
                'validation_split': 0.2,
                'early_stopping_patience': 10
            },
            'isolation_forest': {
                'contamination': 0.1,
                'n_estimators': 100,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        }
    
    @classmethod
    def get_system_metrics(cls) -> list:
        """Get list of system metrics to monitor"""
        return [
            'cpu_percent',
            'memory_percent',
            'memory_available',
            'disk_usage_percent',
            'disk_io_read',
            'disk_io_write',
            'network_io_sent',
            'network_io_recv',
            'boot_time',
            'process_count',
            'load_average_1min',
            'load_average_5min',
            'load_average_15min'
        ]
    
    @classmethod
    def get_paths(cls) -> Dict[str, Path]:
        """Get all important paths"""
        return {
            'base': BASE_DIR,
            'data': DATA_DIR,
            'models': MODELS_DIR,
            'raw_data': RAW_DATA_DIR,
            'processed_data': PROCESSED_DATA_DIR,
            'logs': BASE_DIR / 'logs'
        }

class DevelopmentConfig(Config):
    """Development environment configuration"""
    FLASK_DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production environment configuration"""
    FLASK_DEBUG = False
    LOG_LEVEL = 'WARNING'
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set in production

class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'
    LOG_LEVEL = 'ERROR'

# Configuration selection based on environment
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env: str = None) -> Config:
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'default')
    return config.get(env, config['default'])
'''

with open('ml_system_optimizer/src/config/settings.py', 'w') as f:
    f.write(settings_content)

print("✅ settings.py created successfully!")