# Create setup script
setup_sh_content = '''#!/bin/bash

# ML System Optimizer Setup Script
# This script sets up the complete ML system for system performance optimization

set -e  # Exit on any error

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to create directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p data/models
    mkdir -p logs
    mkdir -p mlruns
    
    print_success "Directories created successfully"
}

# Function to check Python version
check_python() {
    print_status "Checking Python version..."
    
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_status "Found Python $PYTHON_VERSION"
        
        # Check if version is >= 3.8
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
            print_success "Python version is compatible"
        else
            print_error "Python 3.8 or higher is required"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Function to create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_success "Virtual environment setup complete"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Function to setup environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# ML System Optimizer Environment Configuration

# Flask Configuration
FLASK_ENV=development
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
SECRET_KEY=dev-secret-key-change-in-production

# Monitoring Configuration
MONITORING_INTERVAL=30
DATA_COLLECTION_DURATION=3600
DASHBOARD_UPDATE_INTERVAL=10

# Model Configuration
LSTM_LOOKBACK_WINDOW=60
FORECAST_HORIZON=12
MODEL_RETRAIN_INTERVAL=86400
ANOMALY_THRESHOLD=2.5

# System Thresholds
CPU_WARNING_THRESHOLD=80.0
CPU_CRITICAL_THRESHOLD=90.0
MEMORY_WARNING_THRESHOLD=80.0
MEMORY_CRITICAL_THRESHOLD=90.0
DISK_WARNING_THRESHOLD=80.0
DISK_CRITICAL_THRESHOLD=90.0

# MLflow Configuration
MLFLOW_TRACKING_URI=file://./mlruns
MLFLOW_EXPERIMENT_NAME=system_performance_optimization

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/system_optimizer.log

# Database Configuration
DATABASE_URL=sqlite:///system_data.db
EOF
        print_success "Environment file created"
    else
        print_warning "Environment file already exists"
    fi
}

# Function to check system dependencies
check_system_deps() {
    print_status "Checking system dependencies..."
    
    # Check for required system packages
    if command_exists gcc; then
        print_success "GCC compiler found"
    else
        print_warning "GCC compiler not found - may need to install build tools"
    fi
    
    # Check available memory
    if command_exists free; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        print_status "Available memory: ${MEMORY_GB}GB"
        
        if [ "$MEMORY_GB" -lt 2 ]; then
            print_warning "System has less than 2GB RAM - ML training may be slow"
        fi
    fi
    
    # Check disk space
    if command_exists df; then
        DISK_SPACE=$(df -h . | awk 'NR==2{print $4}')
        print_status "Available disk space: $DISK_SPACE"
    fi
}

# Function to run initial setup
run_initial_setup() {
    print_status "Running initial setup..."
    
    # Test basic functionality
    python3 -c "
import sys
sys.path.insert(0, '.')
from src.config.settings import get_config
from src.config.logging_config import setup_logging

config = get_config()
setup_logging(config.LOG_LEVEL, config.LOG_FILE)
print('Configuration loaded successfully')
"
    
    if [ $? -eq 0 ]; then
        print_success "Basic functionality test passed"
    else
        print_error "Basic functionality test failed"
        exit 1
    fi
}

# Function to setup Docker (optional)
setup_docker() {
    if command_exists docker; then
        print_status "Docker found - setting up Docker environment..."
        
        # Build Docker image
        docker build -f docker/Dockerfile -t ml-system-optimizer .
        
        if [ $? -eq 0 ]; then
            print_success "Docker image built successfully"
            print_status "You can now run with: docker-compose -f docker/docker-compose.yml up"
        else
            print_warning "Docker image build failed"
        fi
    else
        print_status "Docker not found - skipping Docker setup"
    fi
}

# Function to display usage information
show_usage() {
    echo ""
    echo "ML System Optimizer Setup Complete!"
    echo "=================================="
    echo ""
    echo "Quick Start:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Start system monitoring: python scripts/monitor_system.py --test"
    echo "3. Train models: python scripts/run_training.py --duration 1"
    echo "4. Start API server: python -m src.api.app"
    echo ""
    echo "Docker Usage:"
    echo "- Build and run: cd docker && docker-compose up"
    echo "- Development mode: cd docker && docker-compose --profile development up"
    echo ""
    echo "API Endpoints (when server is running):"
    echo "- Health check: http://localhost:5000/"
    echo "- Current metrics: http://localhost:5000/api/system/current"
    echo "- Historical data: http://localhost:5000/api/system/history"
    echo "- Predict forecast: http://localhost:5000/api/predict/forecast"
    echo "- Detect anomalies: http://localhost:5000/api/anomaly/detect"
    echo ""
    echo "Monitoring Dashboard: http://localhost:3000 (when using Docker)"
    echo "MLflow UI: http://localhost:5001 (when using Docker)"
    echo ""
}

# Main setup function
main() {
    echo ""
    echo "======================================"
    echo "ML System Optimizer Setup Script"
    echo "======================================"
    echo ""
    
    # Parse command line arguments
    SKIP_VENV=false
    SKIP_DOCKER=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-venv)
                SKIP_VENV=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-venv     Skip virtual environment creation"
                echo "  --skip-docker   Skip Docker setup"
                echo "  -h, --help      Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run setup steps
    check_python
    check_system_deps
    create_directories
    setup_environment
    
    if [ "$SKIP_VENV" = false ]; then
        create_venv
        install_dependencies
        run_initial_setup
    else
        print_warning "Skipping virtual environment setup"
    fi
    
    if [ "$SKIP_DOCKER" = false ]; then
        setup_docker
    else
        print_status "Skipping Docker setup"
    fi
    
    print_success "Setup completed successfully!"
    show_usage
}

# Run main function
main "$@"
'''

# Make the script executable
with open('ml_system_optimizer/scripts/setup.sh', 'w') as f:
    f.write(setup_sh_content)

print("✅ setup.sh created successfully!")