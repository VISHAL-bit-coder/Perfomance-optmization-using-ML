# Create anomaly detection module
anomaly_detector_content = '''"""
Comprehensive anomaly detection for system performance monitoring
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from typing import Dict, List, Tuple, Optional, Any
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from ..config.settings import get_config
from ..config.logging_config import model_logger

class StatisticalAnomalyDetector:
    """
    Statistical methods for anomaly detection
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.thresholds = {}
        self.statistics = {}
    
    def fit(self, X: np.ndarray, contamination: float = 0.1):
        """
        Fit statistical anomaly detector
        
        Args:
            X: Training data
            contamination: Expected proportion of anomalies
        """
        # Calculate statistics for each feature
        self.statistics = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'median': np.median(X, axis=0),
            'q1': np.percentile(X, 25, axis=0),
            'q3': np.percentile(X, 75, axis=0)
        }
        
        # Set thresholds based on z-score and IQR
        z_threshold = self.config.ANOMALY_THRESHOLD
        self.thresholds = {
            'z_score': z_threshold,
            'iqr_multiplier': 1.5
        }
        
        model_logger.logger.info("Statistical anomaly detector fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using statistical methods
        
        Args:
            X: Data to check for anomalies
            
        Returns:
            Binary array (1 for anomaly, 0 for normal)
        """
        if not self.statistics:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        anomalies = np.zeros(len(X))
        
        # Z-score method
        z_scores = np.abs((X - self.statistics['mean']) / self.statistics['std'])
        z_anomalies = np.any(z_scores > self.thresholds['z_score'], axis=1)
        
        # IQR method
        iqr = self.statistics['q3'] - self.statistics['q1']
        lower_bound = self.statistics['q1'] - self.thresholds['iqr_multiplier'] * iqr
        upper_bound = self.statistics['q3'] + self.thresholds['iqr_multiplier'] * iqr
        
        iqr_anomalies = np.any((X < lower_bound) | (X > upper_bound), axis=1)
        
        # Combine methods
        anomalies = z_anomalies | iqr_anomalies
        
        return anomalies.astype(int)
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for each sample
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores
        """
        if not self.statistics:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Use maximum z-score across features as anomaly score
        z_scores = np.abs((X - self.statistics['mean']) / self.statistics['std'])
        anomaly_scores = np.max(z_scores, axis=1)
        
        return anomaly_scores

class AutoencoderAnomalyDetector:
    """
    Autoencoder-based anomaly detection
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
    
    def build_autoencoder(self, input_dim: int, encoding_dim: int = None) -> Sequential:
        """
        Build autoencoder model
        
        Args:
            input_dim: Input dimension
            encoding_dim: Encoding dimension (bottleneck)
            
        Returns:
            Autoencoder model
        """
        encoding_dim = encoding_dim or max(input_dim // 4, 8)
        
        model = Sequential([
            # Encoder
            Dense(input_dim // 2, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(encoding_dim, activation='relu'),
            
            # Decoder
            Dense(input_dim // 2, activation='relu'),
            Dropout(0.2),
            Dense(input_dim, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        
        model_logger.logger.info(f"Built autoencoder with encoding dimension {encoding_dim}")
        return model
    
    def fit(self, X: np.ndarray, validation_split: float = 0.2, 
            epochs: int = 100, batch_size: int = 32):
        """
        Train the autoencoder
        
        Args:
            X: Training data (normal samples only)
            validation_split: Validation split ratio
            epochs: Training epochs
            batch_size: Batch size
        """
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model if not exists
        if self.model is None:
            self.build_autoencoder(X.shape[1])
        
        # Train autoencoder
        history = self.model.fit(
            X_scaled, X_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # Calculate reconstruction errors for threshold setting
        X_pred = self.model.predict(X_scaled)
        reconstruction_errors = np.mean(np.square(X_scaled - X_pred), axis=1)
        
        # Set threshold as 95th percentile of reconstruction errors
        self.threshold = np.percentile(reconstruction_errors, 95)
        
        model_logger.logger.info(f"Autoencoder trained with threshold {self.threshold:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using reconstruction error
        
        Args:
            X: Data to check for anomalies
            
        Returns:
            Binary array (1 for anomaly, 0 for normal)
        """
        if self.model is None or self.threshold is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        X_pred = self.model.predict(X_scaled)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(X_scaled - X_pred), axis=1)
        
        # Classify as anomaly if error exceeds threshold
        anomalies = (reconstruction_errors > self.threshold).astype(int)
        
        return anomalies
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get reconstruction error scores
        
        Args:
            X: Data to score
            
        Returns:
            Reconstruction error scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        X_pred = self.model.predict(X_scaled)
        
        reconstruction_errors = np.mean(np.square(X_scaled - X_pred), axis=1)
        return reconstruction_errors

class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detection methods
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.detectors = {}
        self.weights = {}
        self.fitted = False
    
    def add_detector(self, name: str, detector: Any, weight: float = 1.0):
        """
        Add a detector to the ensemble
        
        Args:
            name: Detector name
            detector: Detector instance
            weight: Weight for this detector in ensemble
        """
        self.detectors[name] = detector
        self.weights[name] = weight
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit all detectors in the ensemble
        
        Args:
            X: Training data
            y: Labels (optional, for supervised methods)
        """
        model_logger.logger.info(f"Fitting ensemble with {len(self.detectors)} detectors")
        
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'fit'):
                    if y is not None and hasattr(detector, 'fit') and 'y' in detector.fit.__code__.co_varnames:
                        detector.fit(X, y)
                    else:
                        detector.fit(X)
                model_logger.logger.info(f"Fitted detector: {name}")
            except Exception as e:
                model_logger.logger.error(f"Error fitting detector {name}: {e}")
        
        self.fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using ensemble voting
        
        Args:
            X: Data to check for anomalies
            
        Returns:
            Binary array (1 for anomaly, 0 for normal)
        """
        if not self.fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        predictions = []
        weights = []
        
        for name, detector in self.detectors.items():
            try:
                pred = detector.predict(X)
                predictions.append(pred)
                weights.append(self.weights[name])
            except Exception as e:
                model_logger.logger.error(f"Error predicting with detector {name}: {e}")
        
        if not predictions:
            raise ValueError("No detectors produced valid predictions")
        
        # Weighted voting
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        weighted_scores = np.average(predictions, axis=0, weights=weights)
        ensemble_predictions = (weighted_scores > 0.5).astype(int)
        
        return ensemble_predictions
    
    def get_anomaly_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get anomaly scores from all detectors
        
        Args:
            X: Data to score
            
        Returns:
            Dictionary of anomaly scores for each detector
        """
        scores = {}
        
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'get_anomaly_scores'):
                    scores[name] = detector.get_anomaly_scores(X)
                elif hasattr(detector, 'decision_function'):
                    scores[name] = -detector.decision_function(X)  # Convert to anomaly scores
                elif hasattr(detector, 'score_samples'):
                    scores[name] = -detector.score_samples(X)  # Convert to anomaly scores
            except Exception as e:
                model_logger.logger.error(f"Error getting scores from detector {name}: {e}")
        
        return scores

class SystemAnomalyDetector:
    """
    Main anomaly detection system for system performance monitoring
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.ensemble = EnsembleAnomalyDetector(config)
        self.setup_detectors()
    
    def setup_detectors(self):
        """Setup all anomaly detectors in the ensemble"""
        
        # Statistical detector
        statistical_detector = StatisticalAnomalyDetector(self.config)
        self.ensemble.add_detector('statistical', statistical_detector, weight=0.3)
        
        # Isolation Forest
        isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.ensemble.add_detector('isolation_forest', isolation_forest, weight=0.3)
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )
        self.ensemble.add_detector('lof', lof, weight=0.2)
        
        # Autoencoder (will be added after first fit)
        autoencoder = AutoencoderAnomalyDetector(self.config)
        self.ensemble.add_detector('autoencoder', autoencoder, weight=0.2)
        
        model_logger.logger.info("Anomaly detection ensemble configured")
    
    def fit(self, X: np.ndarray):
        """
        Fit the anomaly detection system
        
        Args:
            X: Training data (normal system behavior)
        """
        self.ensemble.fit(X)
        model_logger.logger.info("System anomaly detector fitted")
    
    def detect_anomalies(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies in system data
        
        Args:
            X: System metrics data
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Get ensemble predictions
        anomalies = self.ensemble.predict(X)
        
        # Get individual detector scores
        scores = self.ensemble.get_anomaly_scores(X)
        
        # Calculate summary statistics
        anomaly_rate = np.mean(anomalies)
        anomaly_indices = np.where(anomalies == 1)[0]
        
        results = {
            'anomalies': anomalies,
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_rate': anomaly_rate,
            'total_anomalies': len(anomaly_indices),
            'scores': scores
        }
        
        if len(anomaly_indices) > 0:
            model_logger.logger.warning(
                f"Detected {len(anomaly_indices)} anomalies ({anomaly_rate:.2%} of data)"
            )
        
        return results
    
    def analyze_anomalies(self, X: pd.DataFrame, anomaly_indices: List[int]) -> Dict[str, Any]:
        """
        Analyze detected anomalies to understand their characteristics
        
        Args:
            X: Original data with column names
            anomaly_indices: Indices of detected anomalies
            
        Returns:
            Analysis results
        """
        if len(anomaly_indices) == 0:
            return {'message': 'No anomalies to analyze'}
        
        anomaly_data = X.iloc[anomaly_indices]
        normal_data = X.drop(anomaly_indices)
        
        analysis = {
            'anomaly_summary': anomaly_data.describe(),
            'normal_summary': normal_data.describe(),
            'feature_importance': {},
            'temporal_patterns': {}
        }
        
        # Feature importance based on difference from normal
        for col in X.select_dtypes(include=[np.number]).columns:
            if col in anomaly_data.columns:
                normal_mean = normal_data[col].mean()
                anomaly_mean = anomaly_data[col].mean()
                difference = abs(anomaly_mean - normal_mean) / normal_data[col].std()
                analysis['feature_importance'][col] = difference
        
        # Sort features by importance
        analysis['feature_importance'] = dict(
            sorted(analysis['feature_importance'].items(), 
                   key=lambda x: x[1], reverse=True)
        )
        
        # Temporal patterns (if timestamp available)
        if 'timestamp' in X.columns:
            anomaly_times = pd.to_datetime(X.iloc[anomaly_indices]['timestamp'])
            analysis['temporal_patterns'] = {
                'hours': anomaly_times.dt.hour.value_counts().to_dict(),
                'days_of_week': anomaly_times.dt.dayofweek.value_counts().to_dict(),
                'time_ranges': {
                    'business_hours': sum((anomaly_times.dt.hour >= 9) & 
                                        (anomaly_times.dt.hour <= 17)),
                    'off_hours': sum((anomaly_times.dt.hour < 9) | 
                                   (anomaly_times.dt.hour > 17))
                }
            }
        
        return analysis
    
    def save_model(self, filepath: str):
        """
        Save the trained anomaly detection model
        
        Args:
            filepath: Path to save the model
        """
        joblib.dump(self.ensemble, filepath)
        model_logger.logger.info(f"Anomaly detection model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained anomaly detection model
        
        Args:
            filepath: Path to the saved model
        """
        self.ensemble = joblib.load(filepath)
        model_logger.logger.info(f"Anomaly detection model loaded from {filepath}")

# Global anomaly detector instance
anomaly_detector = SystemAnomalyDetector()
'''

with open('ml_system_optimizer/src/models/anomaly_detector.py', 'w') as f:
    f.write(anomaly_detector_content)

print("✅ anomaly_detector.py created successfully!")