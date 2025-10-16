# Create comprehensive README.md
readme_content = '''# ML System Optimizer

A comprehensive machine learning system for optimizing system performance through predictive analysis. This project provides real-time system monitoring, anomaly detection, performance forecasting, and automated optimization recommendations.

## 🚀 Features

### Core Capabilities
- **Real-time System Monitoring**: Continuous monitoring of CPU, memory, disk, and network metrics using psutil
- **Predictive Forecasting**: LSTM-based time series forecasting for system resource usage
- **Anomaly Detection**: Multi-method ensemble approach for detecting system anomalies
- **Performance Optimization**: Automated recommendations for system performance improvements
- **RESTful API**: Complete API for accessing predictions, metrics, and optimization suggestions
- **MLflow Integration**: Comprehensive experiment tracking and model management
- **Docker Support**: Full containerization with docker-compose for easy deployment

### Machine Learning Models
- **LSTM Neural Networks**: Deep learning models for time series forecasting
- **Isolation Forest**: Unsupervised anomaly detection
- **Statistical Methods**: Z-score and IQR-based anomaly detection
- **Autoencoder**: Neural network-based anomaly detection
- **Ensemble Methods**: Combination of multiple detection algorithms

### Monitoring & Visualization
- **Real-time Dashboard**: Live system metrics visualization
- **Historical Analysis**: Long-term trend analysis and reporting
- **Alert System**: Configurable thresholds and notifications
- **Performance Metrics**: Comprehensive system health indicators

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- Docker and Docker Compose (optional)
- Linux/macOS/Windows

## 🛠️ Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd ml_system_optimizer

# Run automated setup
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Activate Environment
```bash
source venv/bin/activate
```

### 3. Test System Monitoring
```bash
python scripts/monitor_system.py --test
```

### 4. Train Initial Models
```bash
python scripts/run_training.py --duration 1
```

### 5. Start API Server
```bash
python -m src.api.app
```

### 6. Test API Endpoints
```bash
# Health check
curl http://localhost:5000/

# Current system metrics
curl http://localhost:5000/api/system/current

# Get forecasting prediction
curl -X POST http://localhost:5000/api/predict/forecast \\
  -H "Content-Type: application/json" \\
  -d '{"metric": "cpu_percent", "steps": 12}'
```

## 🐳 Docker Deployment

### Quick Start with Docker
```bash
cd docker
docker-compose up -d
```

### Development Mode
```bash
cd docker
docker-compose --profile development up
```

### Services Available
- **ML System Optimizer API**: http://localhost:5000
- **MLflow UI**: http://localhost:5001
- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Jupyter Notebook**: http://localhost:8888 (development mode)

## 📖 API Documentation

### System Monitoring Endpoints

#### Get Current Metrics
```http
GET /api/system/current
```

Response:
```json
{
  "timestamp": "2025-01-15T10:30:00",
  "status": "success",
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 62.8,
    "disk_usage_percent": 78.5,
    "process_count": 142
  }
}
```

#### Get Historical Data
```http
GET /api/system/history?hours=24
```

### Prediction Endpoints

#### Forecast System Metrics
```http
POST /api/predict/forecast
Content-Type: application/json

{
  "metric": "cpu_percent",
  "steps": 12
}
```

Response:
```json
{
  "timestamp": "2025-01-15T10:30:00",
  "status": "success",
  "metric": "cpu_percent",
  "forecast": [
    {
      "timestamp": "2025-01-15T10:35:00",
      "predicted_value": 47.3,
      "metric": "cpu_percent"
    }
  ]
}
```

#### Detect Anomalies
```http
POST /api/anomaly/detect
Content-Type: application/json

{
  "hours": 1
}
```

### Model Management Endpoints

#### Train Models
```http
POST /api/models/train
Content-Type: application/json

{
  "duration_hours": 24
}
```

#### Get Model Status
```http
GET /api/models/status
```

### Optimization Endpoints

#### Get Recommendations
```http
GET /api/optimize/recommendations
```

Response:
```json
{
  "timestamp": "2025-01-15T10:30:00",
  "status": "success",
  "recommendations": [
    {
      "type": "cpu_optimization",
      "priority": "medium",
      "message": "High CPU usage detected: 85.2%",
      "suggestions": [
        "Check for resource-intensive processes",
        "Consider scaling up CPU resources"
      ]
    }
  ]
}
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file or set environment variables:

```env
# Flask Configuration
FLASK_ENV=production
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Monitoring Configuration
MONITORING_INTERVAL=30
MODEL_RETRAIN_INTERVAL=86400

# System Thresholds
CPU_WARNING_THRESHOLD=80.0
CPU_CRITICAL_THRESHOLD=90.0
MEMORY_WARNING_THRESHOLD=80.0
MEMORY_CRITICAL_THRESHOLD=90.0

# MLflow Configuration
MLFLOW_TRACKING_URI=file://./mlruns
MLFLOW_EXPERIMENT_NAME=system_performance_optimization
```

### Model Configuration
Models can be configured in `src/config/settings.py`:

```python
LSTM_LOOKBACK_WINDOW = 60  # Time steps to look back
FORECAST_HORIZON = 12      # Steps to forecast ahead
ANOMALY_THRESHOLD = 2.5    # Z-score threshold for anomalies
```

## 📊 Machine Learning Pipeline

### 1. Data Collection
- **psutil**: System metrics collection
- **Real-time monitoring**: Continuous data gathering
- **Data preprocessing**: Feature engineering and cleaning

### 2. Model Training
- **LSTM Networks**: Time series forecasting
- **Anomaly Detection**: Multiple algorithms ensemble
- **MLflow Tracking**: Experiment management

### 3. Model Deployment
- **REST API**: Model serving
- **Real-time Predictions**: Live inference
- **Model Registry**: Version management

### 4. Monitoring & Optimization
- **Performance Tracking**: Model drift detection
- **Automated Retraining**: Scheduled model updates
- **Alert System**: Threshold-based notifications

## 🏗️ Project Structure

```
ml_system_optimizer/
├── src/
│   ├── data_collection/       # System monitoring and data preprocessing
│   │   ├── system_monitor.py  # psutil-based system monitoring
│   │   └── data_preprocessor.py # Feature engineering
│   ├── models/                # Machine learning models
│   │   ├── time_series_forecaster.py # LSTM forecasting
│   │   ├── anomaly_detector.py # Anomaly detection ensemble
│   │   └── model_trainer.py   # Training orchestration
│   ├── api/                   # REST API
│   │   ├── app.py            # Flask application
│   │   └── utils.py          # API utilities
│   ├── config/               # Configuration
│   │   ├── settings.py       # Application settings
│   │   └── logging_config.py # Logging configuration
│   └── monitoring/           # Monitoring dashboard
├── scripts/                  # Utility scripts
│   ├── setup.sh             # Installation script
│   ├── run_training.py      # Training script
│   └── monitor_system.py    # Monitoring service
├── docker/                  # Docker configuration
│   ├── Dockerfile           # Container definition
│   └── docker-compose.yml   # Multi-service setup
├── mlflow/                  # MLflow configuration
├── notebooks/               # Jupyter notebooks
├── tests/                   # Unit tests
└── data/                    # Data storage
    ├── raw/                 # Raw system data
    ├── processed/           # Processed features
    └── models/              # Trained models
```

## 🧪 Testing

### Run Unit Tests
```bash
python -m pytest tests/ -v
```

### Run Integration Tests
```bash
# Test system monitoring
python scripts/monitor_system.py --test

# Test model training
python scripts/run_training.py --duration 1

# Test API endpoints
curl http://localhost:5000/api/system/current
```

## 📈 Performance Metrics

### Model Performance
- **LSTM Forecasting**: MAE < 5%, MAPE < 10%
- **Anomaly Detection**: Precision > 85%, Recall > 80%
- **Response Time**: API endpoints < 200ms
- **Throughput**: 1000+ requests/minute

### System Requirements
- **CPU**: 2+ cores recommended
- **Memory**: 4GB+ RAM
- **Storage**: 10GB+ for data and models
- **Network**: Minimal bandwidth requirements

## 🚨 Monitoring & Alerts

### Alert Types
- **CPU Usage**: Warning at 80%, Critical at 90%
- **Memory Usage**: Warning at 80%, Critical at 90%
- **Disk Usage**: Warning at 80%, Critical at 90%
- **Model Drift**: Automatic detection and retraining
- **API Health**: Endpoint availability monitoring

### Alert Channels
- **Logs**: Structured logging with different levels
- **Webhooks**: Configurable webhook notifications
- **Email**: SMTP-based alert notifications (configurable)

## 🔒 Security

### API Security
- **Rate Limiting**: Configurable request limits
- **Input Validation**: Comprehensive request validation
- **Error Handling**: Secure error responses
- **CORS**: Configurable cross-origin requests

### Data Security
- **Local Storage**: All data stored locally by default
- **Encryption**: Environment variable configuration
- **Access Control**: File system permissions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Run tests: `python -m pytest`
5. Commit changes: `git commit -am 'Add new feature'`
6. Push to branch: `git push origin feature/new-feature`
7. Create Pull Request

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd ml_system_optimizer

# Setup development environment
./scripts/setup.sh --skip-docker

# Install development dependencies
pip install -r requirements-dev.txt

# Run in development mode
export FLASK_ENV=development
python -m src.api.app
```

## 📚 Documentation

### API Documentation
- Swagger/OpenAPI documentation available at `/docs` when server is running
- Postman collection available in `docs/postman/`

### Model Documentation
- Model architecture details in `docs/models/`
- Training procedures in `docs/training/`
- Performance benchmarks in `docs/performance/`

## 🐛 Troubleshooting

### Common Issues

#### Installation Issues
```bash
# Permission denied on setup.sh
chmod +x scripts/setup.sh

# Python module not found
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Dependencies installation failed
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

#### Runtime Issues
```bash
# Port already in use
export FLASK_PORT=5001

# Insufficient memory
# Reduce batch size in config/settings.py
LSTM_BATCH_SIZE = 16

# Database locked
rm system_data.db  # Will be recreated
```

#### Docker Issues
```bash
# Docker build failed
docker system prune -f
docker-compose build --no-cache

# Container startup issues
docker-compose logs ml-optimizer
```

### Logging
Logs are available in:
- **Application logs**: `logs/system_optimizer.log`
- **Docker logs**: `docker-compose logs [service_name]`
- **MLflow logs**: Available in MLflow UI

## 📊 Monitoring Dashboard

### Grafana Dashboard
Access at http://localhost:3000 (when using Docker)
- Username: admin
- Password: admin123

### Custom Metrics
- System performance trends
- Model accuracy over time
- API request metrics
- Anomaly detection results

## 🔄 Automated Operations

### Scheduled Tasks
- **Model Retraining**: Every 24 hours (configurable)
- **Data Cleanup**: Weekly cleanup of old data
- **Health Checks**: Continuous monitoring
- **Performance Evaluation**: Daily model performance checks

### CI/CD Integration
```yaml
# Example GitHub Actions workflow
name: ML System Optimizer CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **psutil**: System monitoring capabilities
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Machine learning algorithms
- **Flask**: Web framework
- **MLflow**: Experiment tracking
- **Docker**: Containerization
- **Grafana**: Monitoring dashboards

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

## 🗺️ Roadmap

### Version 2.0
- [ ] Multi-node system monitoring
- [ ] Advanced neural architectures (Transformers)
- [ ] Kubernetes integration
- [ ] Advanced alerting system
- [ ] Web-based configuration UI

### Version 2.1
- [ ] Cloud provider integrations (AWS, GCP, Azure)
- [ ] Mobile app for monitoring
- [ ] Advanced optimization algorithms
- [ ] Predictive scaling recommendations

---

**ML System Optimizer** - Intelligent system performance optimization through machine learning.
'''

with open('ml_system_optimizer/README.md', 'w') as f:
    f.write(readme_content)

print("✅ README.md created successfully!")