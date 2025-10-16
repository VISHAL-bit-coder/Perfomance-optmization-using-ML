# Create LSTM time series forecaster
time_series_forecaster_content = '''"""
LSTM-based time series forecasting for system performance prediction
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import joblib
from pathlib import Path
import mlflow
import mlflow.keras

from ..config.settings import get_config
from ..config.logging_config import model_logger

class LSTMForecaster:
    """
    LSTM-based time series forecaster for system performance metrics
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.history = None
        self.scaler = None
        self.feature_columns = []
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def build_model(self, input_shape: Tuple[int, int], 
                   output_size: int = 1,
                   lstm_units: List[int] = None,
                   dropout_rate: float = 0.2) -> Sequential:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            output_size: Number of output predictions
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled LSTM model
        """
        lstm_units = lstm_units or self.config.get_model_config()['lstm']['units']
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            lstm_units[0], 
            return_sequences=True if len(lstm_units) > 1 else False,
            input_shape=input_shape
        ))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:], 1):
            return_seq = i < len(lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_seq))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
        
        # Dense layers for output
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_size, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        model_logger.logger.info(f"Built LSTM model with architecture: {[layer.output_shape for layer in model.layers]}")
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = None, batch_size: int = None,
              verbose: int = 1) -> Dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        epochs = epochs or self.config.get_model_config()['lstm']['epochs']
        batch_size = batch_size or self.config.get_model_config()['lstm']['batch_size']
        
        model_logger.log_training_start('LSTM', len(X_train))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config.get_model_config()['lstm']['early_stopping_patience'],
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Model checkpoint
        model_path = self.config.get_paths()['models'] / 'lstm_checkpoint.h5'
        callbacks.append(ModelCheckpoint(
            str(model_path),
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True
        ))
        
        # Validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train model
        start_time = pd.Timestamp.now()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Get final metrics
        final_metrics = {
            'final_loss': self.history.history['loss'][-1],
            'final_mae': self.history.history['mae'][-1],
            'final_mape': self.history.history['mape'][-1]
        }
        
        if validation_data:
            final_metrics.update({
                'final_val_loss': self.history.history['val_loss'][-1],
                'final_val_mae': self.history.history['val_mae'][-1],
                'final_val_mape': self.history.history['val_mape'][-1]
            })
        
        model_logger.log_training_complete('LSTM', training_time, final_metrics)
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input features for prediction
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        model_logger.log_prediction('LSTM', X.shape, predictions.shape)
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            # Multi-step prediction
            metrics = {}
            for i in range(y_test.shape[1]):
                metrics[f'mse_step_{i+1}'] = mean_squared_error(y_test[:, i], y_pred[:, i])
                metrics[f'mae_step_{i+1}'] = mean_absolute_error(y_test[:, i], y_pred[:, i])
                metrics[f'r2_step_{i+1}'] = r2_score(y_test[:, i], y_pred[:, i])
            
            # Overall metrics
            metrics['overall_mse'] = mean_squared_error(y_test.flatten(), y_pred.flatten())
            metrics['overall_mae'] = mean_absolute_error(y_test.flatten(), y_pred.flatten())
            metrics['overall_r2'] = r2_score(y_test.flatten(), y_pred.flatten())
        else:
            # Single-step prediction
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
        
        model_logger.log_model_performance('LSTM', metrics)
        return metrics
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        
        # MAPE
        axes[1, 0].plot(self.history.history['mape'], label='Training MAPE')
        if 'val_mape' in self.history.history:
            axes[1, 0].plot(self.history.history['val_mape'], label='Validation MAPE')
        axes[1, 0].set_title('Mean Absolute Percentage Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAPE')
        axes[1, 0].legend()
        
        # Learning rate (if available)
        if hasattr(self.model.optimizer, 'learning_rate'):
            axes[1, 1].plot(range(len(self.history.history['loss'])), 
                           [float(self.model.optimizer.learning_rate)] * len(self.history.history['loss']))
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            model_logger.logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, X_test: np.ndarray, y_test: np.ndarray, 
                        n_samples: int = 100, save_path: str = None):
        """
        Plot predictions vs actual values
        
        Args:
            X_test: Test features
            y_test: Test targets
            n_samples: Number of samples to plot
            save_path: Path to save the plot
        """
        y_pred = self.predict(X_test)
        
        # Select subset of samples
        indices = np.random.choice(len(y_test), min(n_samples, len(y_test)), replace=False)
        y_test_subset = y_test[indices]
        y_pred_subset = y_pred[indices]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time series plot
        axes[0].plot(y_test_subset[:50], label='Actual', alpha=0.7)
        axes[0].plot(y_pred_subset[:50], label='Predicted', alpha=0.7)
        axes[0].set_title('Predictions vs Actual (First 50 samples)')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        
        # Scatter plot
        axes[1].scatter(y_test_subset, y_pred_subset, alpha=0.5)
        axes[1].plot([y_test_subset.min(), y_test_subset.max()], 
                    [y_test_subset.min(), y_test_subset.max()], 'r--', lw=2)
        axes[1].set_title('Predicted vs Actual')
        axes[1].set_xlabel('Actual')
        axes[1].set_ylabel('Predicted')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        model_logger.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = load_model(filepath)
        model_logger.logger.info(f"Model loaded from {filepath}")
    
    def forecast_future(self, last_sequence: np.ndarray, 
                       n_steps: int) -> np.ndarray:
        """
        Forecast future values using the last known sequence
        
        Args:
            last_sequence: Last known sequence of data
            n_steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_steps):
            # Predict next step
            next_pred = self.model.predict(current_sequence.reshape(1, *current_sequence.shape))
            forecasts.append(next_pred[0])
            
            # Update sequence (remove first timestep, add prediction)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_pred[0]
        
        return np.array(forecasts)

# Global forecaster instance
lstm_forecaster = LSTMForecaster()
'''

with open('ml_system_optimizer/src/models/time_series_forecaster.py', 'w') as f:
    f.write(time_series_forecaster_content)

print("✅ time_series_forecaster.py created successfully!")