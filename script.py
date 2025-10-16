import os
import json

# Create comprehensive project structure
project_structure = {
    'ml_system_optimizer': {
        'src': {
            'data_collection': {
                '__init__.py': '',
                'system_monitor.py': '',
                'data_preprocessor.py': ''
            },
            'models': {
                '__init__.py': '',
                'time_series_forecaster.py': '',
                'anomaly_detector.py': '',
                'model_trainer.py': ''
            },
            'api': {
                '__init__.py': '',
                'app.py': '',
                'routes.py': '',
                'utils.py': ''
            },
            'monitoring': {
                '__init__.py': '',
                'dashboard.py': '',
                'metrics_collector.py': ''
            },
            'config': {
                '__init__.py': '',
                'settings.py': '',
                'logging_config.py': ''
            },
            'utils': {
                '__init__.py': '',
                'helpers.py': '',
                'data_utils.py': ''
            }
        },
        'tests': {
            '__init__.py': '',
            'test_models.py': '',
            'test_api.py': '',
            'test_data_collection.py': ''
        },
        'notebooks': {
            'exploratory_analysis.ipynb': '',
            'model_development.ipynb': '',
            'performance_analysis.ipynb': ''
        },
        'data': {
            'raw': {},
            'processed': {},
            'models': {}
        },
        'docker': {
            'Dockerfile': '',
            'docker-compose.yml': '',
            'requirements.txt': ''
        },
        'mlflow': {
            'mlflow_setup.py': '',
            'model_registry.py': ''
        },
        'scripts': {
            'setup.sh': '',
            'run_training.py': '',
            'deploy_model.py': '',
            'monitor_system.py': ''
        },
        'requirements.txt': '',
        'README.md': '',
        'setup.py': '',
        '.env': '',
        '.gitignore': ''
    }
}

def create_directory_structure(structure, base_path=''):
    """Recursively create directory structure"""
    for name, content in structure.items():
        current_path = os.path.join(base_path, name)
        
        if isinstance(content, dict):
            # It's a directory
            os.makedirs(current_path, exist_ok=True)
            create_directory_structure(content, current_path)
        else:
            # It's a file
            os.makedirs(os.path.dirname(current_path), exist_ok=True)
            with open(current_path, 'w') as f:
                f.write(content)

# Create the project structure
create_directory_structure(project_structure)

print("✅ Project directory structure created successfully!")
print("\n📁 Project Structure:")
for root, dirs, files in os.walk('ml_system_optimizer'):
    level = root.replace('ml_system_optimizer', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{subindent}{file}')