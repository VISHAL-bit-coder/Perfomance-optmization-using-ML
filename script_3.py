# Create comprehensive logging configuration
logging_config_content = '''"""
Logging configuration for ML System Optimizer
"""
import logging
import logging.handlers
import os
from pathlib import Path
import colorlog
from datetime import datetime

def setup_logging(log_level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(exist_ok=True, parents=True)
    
    # Configure root logger
    logger = logging.getLogger('ml_system_optimizer')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (optional)
    
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f'ml_system_optimizer.{name}')
    return logging.getLogger('ml_system_optimizer')

class MLFlowLogger:
    """Custom logger for MLflow experiments"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.logger = get_logger('mlflow')
    
    def log_params(self, params: dict):
        """Log parameters"""
        self.logger.info(f"Parameters logged for {self.experiment_name}: {params}")
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics"""
        step_info = f" at step {step}" if step else ""
        self.logger.info(f"Metrics logged for {self.experiment_name}{step_info}: {metrics}")
    
    def log_model_performance(self, model_name: str, performance_metrics: dict):
        """Log model performance"""
        self.logger.info(f"Model {model_name} performance: {performance_metrics}")

class SystemMonitorLogger:
    """Custom logger for system monitoring"""
    
    def __init__(self):
        self.logger = get_logger('system_monitor')
    
    def log_system_metrics(self, metrics: dict):
        """Log system metrics"""
        self.logger.debug(f"System metrics: {metrics}")
    
    def log_anomaly_detected(self, metric_name: str, value: float, threshold: float):
        """Log anomaly detection"""
        self.logger.warning(
            f"Anomaly detected - {metric_name}: {value:.2f} (threshold: {threshold:.2f})"
        )
    
    def log_performance_optimization(self, optimization_type: str, details: dict):
        """Log performance optimization actions"""
        self.logger.info(
            f"Performance optimization applied - {optimization_type}: {details}"
        )

class APILogger:
    """Custom logger for API requests"""
    
    def __init__(self):
        self.logger = get_logger('api')
    
    def log_request(self, method: str, endpoint: str, status_code: int, response_time: float):
        """Log API request"""
        self.logger.info(
            f"{method} {endpoint} - {status_code} - {response_time:.3f}s"
        )
    
    def log_error(self, endpoint: str, error: str):
        """Log API error"""
        self.logger.error(f"API Error at {endpoint}: {error}")

class ModelLogger:
    """Custom logger for model training and prediction"""
    
    def __init__(self):
        self.logger = get_logger('model')
    
    def log_training_start(self, model_name: str, dataset_size: int):
        """Log training start"""
        self.logger.info(
            f"Training started for {model_name} with {dataset_size} samples"
        )
    
    def log_training_complete(self, model_name: str, duration: float, final_metrics: dict):
        """Log training completion"""
        self.logger.info(
            f"Training completed for {model_name} in {duration:.2f}s - Metrics: {final_metrics}"
        )
    
    def log_prediction(self, model_name: str, input_shape: tuple, prediction_shape: tuple):
        """Log prediction"""
        self.logger.debug(
            f"Prediction made by {model_name} - Input: {input_shape}, Output: {prediction_shape}"
        )
    
    def log_model_drift(self, model_name: str, drift_score: float, threshold: float):
        """Log model drift detection"""
        self.logger.warning(
            f"Model drift detected for {model_name} - Score: {drift_score:.4f} (threshold: {threshold:.4f})"
        )

# Performance monitoring decorator
def log_performance(logger_name: str = None):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or 'performance')
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"{func.__name__} completed in {duration:.4f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"{func.__name__} failed after {duration:.4f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator

# Global loggers for easy access
system_logger = SystemMonitorLogger()
api_logger = APILogger()
model_logger = ModelLogger()
'''

with open('ml_system_optimizer/src/config/logging_config.py', 'w') as f:
    f.write(logging_config_content)

print("✅ logging_config.py created successfully!")