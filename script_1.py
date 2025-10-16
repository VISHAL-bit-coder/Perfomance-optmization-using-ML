# Create comprehensive requirements.txt
requirements_content = """# Core ML libraries
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
tensorflow==2.13.0
keras==2.13.1

# Time series and forecasting
statsmodels==0.14.0
prophet==1.1.4
tsfresh==0.20.0

# System monitoring
psutil==5.9.5
GPUtil==1.4.0

# Web framework and API
Flask==2.3.2
Flask-CORS==4.0.0
Flask-RESTful==0.3.10
gunicorn==21.2.0

# MLflow for model management
mlflow==2.5.0
mlflow[extras]==2.5.0

# Data visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# Database
SQLAlchemy==2.0.19
sqlite3

# Monitoring and logging
prometheus-client==0.17.1
grafana-api==1.0.3

# Utilities
python-dotenv==1.0.0
joblib==1.3.1
pyyaml==6.0.1
requests==2.31.0

# Development and testing
pytest==7.4.0
pytest-cov==4.1.0
black==23.7.0
flake8==6.0.0

# Async and scheduling
celery==5.3.1
redis==4.6.0
APScheduler==3.10.4

# Docker and containerization
docker==6.1.3

# Additional utilities
tqdm==4.65.0
click==8.1.6
colorlog==6.7.0
"""

with open('ml_system_optimizer/requirements.txt', 'w') as f:
    f.write(requirements_content)

print("✅ requirements.txt created successfully!")