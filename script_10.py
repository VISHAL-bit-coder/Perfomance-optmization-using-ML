# Create Flask API application
flask_app_content = '''"""
Flask API for ML System Optimizer
"""
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, Any, List
import logging

from ..config.settings import get_config
from ..config.logging_config import setup_logging, api_logger
from ..data_collection.system_monitor import SystemMonitor
from ..models.model_trainer import ModelTrainer
from .utils import validate_request_data, format_response, handle_errors

def create_app(config=None):
    """
    Create and configure Flask application
    
    Args:
        config: Configuration object
    
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    
    # Configuration
    config = config or get_config()
    app.config.from_object(config)
    
    # Setup CORS
    CORS(app)
    
    # Setup logging
    setup_logging(config.LOG_LEVEL, config.LOG_FILE)
    
    # Initialize components
    system_monitor = SystemMonitor(config)
    model_trainer = ModelTrainer(config)
    
    # Global variables for background monitoring
    monitoring_thread = None
    monitoring_active = False
    trained_models = {}
    
    def start_background_monitoring():
        """Start background system monitoring"""
        nonlocal monitoring_active
        monitoring_active = True
        system_monitor.start_monitoring(config.MONITORING_INTERVAL)
        api_logger.logger.info("Background monitoring started")
    
    def load_models():
        """Load trained models"""
        nonlocal trained_models
        try:
            trained_models = model_trainer.load_trained_models()
            api_logger.logger.info(f"Loaded {len(trained_models)} models")
        except Exception as e:
            api_logger.logger.error(f"Error loading models: {e}")
    
    # Initialize models and monitoring
    load_models()
    start_background_monitoring()
    
    @app.route('/', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return format_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'monitoring_active': monitoring_active,
            'models_loaded': len(trained_models)
        })
    
    @app.route('/api/system/current', methods=['GET'])
    @handle_errors
    def get_current_metrics():
        """Get current system metrics"""
        try:
            summary = system_monitor.get_system_summary()
            api_logger.log_request('GET', '/api/system/current', 200, 0.0)
            return format_response(summary)
        except Exception as e:
            api_logger.log_error('/api/system/current', str(e))
            return format_response({'error': str(e)}, status_code=500)
    
    @app.route('/api/system/history', methods=['GET'])
    @handle_errors
    def get_historical_metrics():
        """Get historical system metrics"""
        try:
            hours = request.args.get('hours', 24, type=int)
            
            if hours <= 0 or hours > 168:  # Max 1 week
                return format_response({'error': 'Hours must be between 1 and 168'}, status_code=400)
            
            df = system_monitor.get_historical_data(hours=hours)
            
            if df.empty:
                return format_response({'data': [], 'message': 'No historical data available'})
            
            # Convert to JSON-serializable format
            data = df.to_dict('records')
            for record in data:
                record['timestamp'] = record['timestamp'].isoformat() if pd.notna(record['timestamp']) else None
            
            api_logger.log_request('GET', '/api/system/history', 200, 0.0)
            return format_response({
                'data': data,
                'total_records': len(data),
                'time_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                }
            })
        except Exception as e:
            api_logger.log_error('/api/system/history', str(e))
            return format_response({'error': str(e)}, status_code=500)
    
    @app.route('/api/predict/forecast', methods=['POST'])
    @handle_errors
    def predict_forecast():
        """Predict future system metrics"""
        try:
            data = request.get_json()
            
            if not data:
                return format_response({'error': 'No data provided'}, status_code=400)
            
            metric = data.get('metric', 'cpu_percent')
            steps = data.get('steps', 12)
            
            if metric not in ['cpu_percent', 'memory_percent']:
                return format_response({'error': 'Unsupported metric'}, status_code=400)
            
            if steps <= 0 or steps > 100:
                return format_response({'error': 'Steps must be between 1 and 100'}, status_code=400)
            
            # Check if model is available
            model_key = f'forecasting_{metric}'
            if model_key not in trained_models:
                return format_response({'error': f'Model for {metric} not available'}, status_code=404)
            
            # Get recent data for prediction
            df = system_monitor.get_historical_data(hours=2)  # Get 2 hours of recent data
            
            if df.empty:
                return format_response({'error': 'Insufficient historical data for prediction'}, status_code=400)
            
            # Prepare data and make prediction
            model_info = trained_models[model_key]
            forecaster = model_info['model']
            preprocessor = model_info['preprocessor']
            
            # Create features
            df_processed = preprocessor.create_features(df)
            
            # Get last sequence for prediction
            lookback_window = config.LSTM_LOOKBACK_WINDOW
            if len(df_processed) < lookback_window:
                return format_response({'error': 'Insufficient data for prediction'}, status_code=400)
            
            # Prepare sequence
            numeric_features = df_processed.select_dtypes(include=[np.number]).columns
            numeric_features = [col for col in numeric_features if col not in ['timestamp']]
            
            last_sequence = df_processed[numeric_features].tail(lookback_window).values
            last_sequence_scaled = preprocessor.scalers['standard'].transform(last_sequence)
            
            # Make forecast
            forecast = forecaster.forecast_future(last_sequence_scaled, steps)
            
            # Generate timestamps for forecast
            last_timestamp = df['timestamp'].max()
            forecast_timestamps = [
                (last_timestamp + timedelta(minutes=5*i)).isoformat() 
                for i in range(1, steps+1)
            ]
            
            forecast_data = [
                {'timestamp': ts, 'predicted_value': float(val[0]), 'metric': metric}
                for ts, val in zip(forecast_timestamps, forecast)
            ]
            
            api_logger.log_request('POST', '/api/predict/forecast', 200, 0.0)
            return format_response({
                'metric': metric,
                'forecast': forecast_data,
                'metadata': {
                    'steps': steps,
                    'model_type': 'LSTM',
                    'prediction_timestamp': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            api_logger.log_error('/api/predict/forecast', str(e))
            return format_response({'error': str(e)}, status_code=500)
    
    @app.route('/api/anomaly/detect', methods=['POST'])
    @handle_errors
    def detect_anomalies():
        """Detect anomalies in system metrics"""
        try:
            data = request.get_json()
            hours = data.get('hours', 1) if data else 1
            
            if hours <= 0 or hours > 24:
                return format_response({'error': 'Hours must be between 1 and 24'}, status_code=400)
            
            # Check if anomaly detection model is available
            if 'anomaly_detection' not in trained_models:
                return format_response({'error': 'Anomaly detection model not available'}, status_code=404)
            
            # Get recent data
            df = system_monitor.get_historical_data(hours=hours)
            
            if df.empty:
                return format_response({'error': 'No data available for anomaly detection'}, status_code=400)
            
            # Prepare data
            anomaly_detector = trained_models['anomaly_detection']
            df_processed = anomaly_detector.preprocessor.create_anomaly_detection_features(df)
            
            # Select numeric features
            numeric_features = df_processed.select_dtypes(include=[np.number]).columns
            numeric_features = [col for col in numeric_features if col not in ['timestamp']]
            
            X = df_processed[numeric_features].values
            X = np.nan_to_num(X, nan=0.0)
            
            # Scale features
            X_scaled = anomaly_detector.ensemble.preprocessor.scalers['standard'].transform(X)
            
            # Detect anomalies
            anomaly_results = anomaly_detector.detect_anomalies(X_scaled)
            
            # Analyze anomalies
            analysis = anomaly_detector.analyze_anomalies(df_processed, anomaly_results['anomaly_indices'])
            
            # Format response
            response_data = {
                'anomaly_rate': anomaly_results['anomaly_rate'],
                'total_anomalies': anomaly_results['total_anomalies'],
                'anomaly_timestamps': [
                    df.iloc[idx]['timestamp'].isoformat() 
                    for idx in anomaly_results['anomaly_indices']
                ],
                'analysis': analysis,
                'metadata': {
                    'detection_timestamp': datetime.now().isoformat(),
                    'data_period_hours': hours,
                    'samples_analyzed': len(df)
                }
            }
            
            api_logger.log_request('POST', '/api/anomaly/detect', 200, 0.0)
            return format_response(response_data)
            
        except Exception as e:
            api_logger.log_error('/api/anomaly/detect', str(e))
            return format_response({'error': str(e)}, status_code=500)
    
    @app.route('/api/models/train', methods=['POST'])
    @handle_errors
    def train_models():
        """Train or retrain models"""
        try:
            data = request.get_json()
            duration_hours = data.get('duration_hours', 24) if data else 24
            
            if duration_hours <= 0 or duration_hours > 168:  # Max 1 week
                return format_response({'error': 'Duration must be between 1 and 168 hours'}, status_code=400)
            
            # Start training in background
            def background_training():
                try:
                    results = model_trainer.full_training_pipeline(duration_hours)
                    # Reload models after training
                    nonlocal trained_models
                    trained_models = model_trainer.load_trained_models()
                    api_logger.logger.info("Model training completed and models reloaded")
                except Exception as e:
                    api_logger.logger.error(f"Background training failed: {e}")
            
            training_thread = threading.Thread(target=background_training)
            training_thread.start()
            
            api_logger.log_request('POST', '/api/models/train', 202, 0.0)
            return format_response({
                'message': 'Model training started',
                'status': 'training',
                'duration_hours': duration_hours,
                'estimated_completion': (datetime.now() + timedelta(minutes=10)).isoformat()
            }, status_code=202)
            
        except Exception as e:
            api_logger.log_error('/api/models/train', str(e))
            return format_response({'error': str(e)}, status_code=500)
    
    @app.route('/api/models/status', methods=['GET'])
    @handle_errors
    def get_model_status():
        """Get status of trained models"""
        try:
            status = {
                'models_available': list(trained_models.keys()),
                'total_models': len(trained_models),
                'last_check': datetime.now().isoformat()
            }
            
            # Check for training metadata
            metadata_path = config.get_paths()['models'] / 'training_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                status['last_training'] = metadata.get('training_date')
                status['training_config'] = metadata.get('config', {})
            
            api_logger.log_request('GET', '/api/models/status', 200, 0.0)
            return format_response(status)
            
        except Exception as e:
            api_logger.log_error('/api/models/status', str(e))
            return format_response({'error': str(e)}, status_code=500)
    
    @app.route('/api/system/stream', methods=['GET'])
    def stream_metrics():
        """Server-sent events endpoint for real-time metrics"""
        def generate():
            while True:
                try:
                    summary = system_monitor.get_system_summary()
                    yield f"data: {json.dumps(summary)}\\n\\n"
                    time.sleep(config.DASHBOARD_UPDATE_INTERVAL)
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\\n\\n"
                    break
        
        return Response(generate(), mimetype='text/plain')
    
    @app.route('/api/optimize/recommendations', methods=['GET'])
    @handle_errors
    def get_optimization_recommendations():
        """Get system optimization recommendations"""
        try:
            # Get current metrics
            current_metrics = system_monitor.collect_metrics()
            
            recommendations = []
            
            # CPU optimization
            if current_metrics.cpu_percent > config.CPU_WARNING_THRESHOLD:
                recommendations.append({
                    'type': 'cpu_optimization',
                    'priority': 'high' if current_metrics.cpu_percent > config.CPU_CRITICAL_THRESHOLD else 'medium',
                    'message': f'High CPU usage detected: {current_metrics.cpu_percent:.1f}%',
                    'suggestions': [
                        'Check for resource-intensive processes',
                        'Consider scaling up CPU resources',
                        'Optimize application performance'
                    ]
                })
            
            # Memory optimization
            if current_metrics.memory_percent > config.MEMORY_WARNING_THRESHOLD:
                recommendations.append({
                    'type': 'memory_optimization',
                    'priority': 'high' if current_metrics.memory_percent > config.MEMORY_CRITICAL_THRESHOLD else 'medium',
                    'message': f'High memory usage detected: {current_metrics.memory_percent:.1f}%',
                    'suggestions': [
                        'Check for memory leaks',
                        'Consider increasing available memory',
                        'Optimize memory usage patterns'
                    ]
                })
            
            # Disk optimization
            if current_metrics.disk_usage_percent > config.DISK_WARNING_THRESHOLD:
                recommendations.append({
                    'type': 'disk_optimization',
                    'priority': 'high' if current_metrics.disk_usage_percent > config.DISK_CRITICAL_THRESHOLD else 'medium',
                    'message': f'High disk usage detected: {current_metrics.disk_usage_percent:.1f}%',
                    'suggestions': [
                        'Clean up unnecessary files',
                        'Archive old data',
                        'Consider adding more storage'
                    ]
                })
            
            if not recommendations:
                recommendations.append({
                    'type': 'system_healthy',
                    'priority': 'info',
                    'message': 'System is running within normal parameters',
                    'suggestions': ['Continue monitoring for optimal performance']
                })
            
            api_logger.log_request('GET', '/api/optimize/recommendations', 200, 0.0)
            return format_response({
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat(),
                'system_status': 'healthy' if len(recommendations) == 1 and recommendations[0]['type'] == 'system_healthy' else 'needs_attention'
            })
            
        except Exception as e:
            api_logger.log_error('/api/optimize/recommendations', str(e))
            return format_response({'error': str(e)}, status_code=500)
    
    @app.errorhandler(404)
    def not_found(error):
        return format_response({'error': 'Endpoint not found'}, status_code=404)
    
    @app.errorhandler(500)
    def internal_error(error):
        return format_response({'error': 'Internal server error'}, status_code=500)
    
    # Cleanup on shutdown
    @app.teardown_appcontext
    def cleanup(exception):
        if monitoring_active:
            system_monitor.stop_monitoring()
    
    return app

# Create the Flask app
app = create_app()

if __name__ == '__main__':
    config = get_config()
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
'''

with open('ml_system_optimizer/src/api/app.py', 'w') as f:
    f.write(flask_app_content)

print("✅ app.py created successfully!")