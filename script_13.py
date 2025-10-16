# Create docker-compose.yml
docker_compose_content = '''version: '3.8'

services:
  # Main ML System Optimizer API
  ml-optimizer:
    build:
      context: ..
      dockerfile: docker/Dockerfile
      target: production
    container_name: ml_system_optimizer
    ports:
      - "5000:5000"
    volumes:
      - ml_data:/app/data
      - ml_logs:/app/logs
      - ml_models:/app/data/models
    environment:
      - FLASK_ENV=production
      - MONITORING_INTERVAL=30
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - DATABASE_URL=sqlite:///app/data/system_data.db
    depends_on:
      - redis
      - mlflow
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - ml_network

  # MLflow for experiment tracking
  mlflow:
    image: python:3.9-slim
    container_name: mlflow_server
    ports:
      - "5001:5000"
    volumes:
      - mlflow_data:/mlflow
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    command: >
      bash -c "
        pip install mlflow[extras]==2.5.0 &&
        mkdir -p /mlflow/artifacts &&
        mlflow server 
        --backend-store-uri sqlite:///mlflow/mlflow.db 
        --default-artifact-root /mlflow/artifacts 
        --host 0.0.0.0 
        --port 5000
      "
    restart: unless-stopped
    networks:
      - ml_network

  # Redis for caching and task queue
  redis:
    image: redis:7-alpine
    container_name: redis_cache
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - ml_network

  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=15d'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - ml_network

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - ml_network

  # Jupyter Notebook for development (optional)
  jupyter:
    build:
      context: ..
      dockerfile: docker/Dockerfile
      target: development
    container_name: jupyter_notebook
    ports:
      - "8888:8888"
    volumes:
      - ml_data:/app/data
      - ml_notebooks:/app/notebooks
      - ../notebooks:/app/notebooks
    environment:
      - JUPYTER_ENABLE_LAB=yes
    command: >
      bash -c "
        jupyter lab 
        --ip=0.0.0.0 
        --port=8888 
        --no-browser 
        --allow-root 
        --NotebookApp.token='' 
        --NotebookApp.password=''
      "
    restart: unless-stopped
    networks:
      - ml_network
    profiles:
      - development

  # System monitoring dashboard
  dashboard:
    build:
      context: ..
      dockerfile: docker/Dockerfile
      target: development
    container_name: system_dashboard
    ports:
      - "8080:8080"
    volumes:
      - ml_data:/app/data
    environment:
      - FLASK_APP=src.monitoring.dashboard
      - FLASK_ENV=development
    command: python -m src.monitoring.dashboard
    depends_on:
      - ml-optimizer
    restart: unless-stopped
    networks:
      - ml_network
    profiles:
      - development

volumes:
  ml_data:
    driver: local
  ml_logs:
    driver: local
  ml_models:
    driver: local
  ml_notebooks:
    driver: local
  mlflow_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  ml_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

# Override for development
# To use: docker-compose --profile development up
'''

with open('ml_system_optimizer/docker/docker-compose.yml', 'w') as f:
    f.write(docker_compose_content)

print("✅ docker-compose.yml created successfully!")