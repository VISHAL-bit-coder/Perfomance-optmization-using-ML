# Create MLflow setup module
mlflow_setup_content = '''"""
MLflow setup and configuration for experiment tracking
"""
import mlflow
import mlflow.sklearn
import mlflow.keras
import mlflow.tracking
from mlflow.tracking import MlflowClient
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import numpy as np

from ..config.settings import get_config
from ..config.logging_config import get_logger

logger = get_logger('mlflow')

class MLflowManager:
    """
    Comprehensive MLflow management for experiment tracking and model registry
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.client = None
        self.experiment_id = None
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking URI and experiment"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
            self.client = MlflowClient()
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.config.MLFLOW_EXPERIMENT_NAME)
                if experiment is None:
                    self.experiment_id = mlflow.create_experiment(
                        self.config.MLFLOW_EXPERIMENT_NAME,
                        tags={
                            "project": "ML System Optimizer",
                            "version": "1.0.0",
                            "created_date": datetime.now().isoformat()
                        }
                    )
                else:
                    self.experiment_id = experiment.experiment_id
                
                mlflow.set_experiment(self.config.MLFLOW_EXPERIMENT_NAME)
                logger.info(f"MLflow experiment setup complete: {self.config.MLFLOW_EXPERIMENT_NAME}")
                
            except Exception as e:
                logger.error(f"Error setting up MLflow experiment: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            raise
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for the run
            tags: Additional tags for the run
        
        Returns:
            MLflow active run
        """
        run_tags = {
            "framework": "scikit-learn/tensorflow",
            "environment": os.getenv('FLASK_ENV', 'development'),
            "timestamp": datetime.now().isoformat()
        }
        
        if tags:
            run_tags.update(tags)
        
        run = mlflow.start_run(run_name=run_name, tags=run_tags)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow
        
        Args:
            params: Parameters dictionary
        """
        try:
            # Convert complex types to strings
            cleaned_params = {}
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    cleaned_params[key] = json.dumps(value)
                elif isinstance(value, np.ndarray):
                    cleaned_params[key] = str(value.shape)
                else:
                    cleaned_params[key] = str(value)
            
            mlflow.log_params(cleaned_params)
            logger.debug(f"Logged {len(params)} parameters")
            
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics to MLflow
        
        Args:
            metrics: Metrics dictionary
            step: Step number (optional)
        """
        try:
            for key, value in metrics.items():
                # Ensure value is numeric
                if isinstance(value, (int, float, np.integer, np.floating)):
                    mlflow.log_metric(key, float(value), step)
                else:
                    logger.warning(f"Skipping non-numeric metric: {key} = {value}")
            
            logger.debug(f"Logged {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Log artifact to MLflow
        
        Args:
            local_path: Path to local file
            artifact_path: Path in artifact store
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
            
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")
    
    def log_model(self, model, model_name: str, 
                  framework: str = "sklearn",
                  signature=None, 
                  input_example=None,
                  registered_model_name: str = None):
        """
        Log model to MLflow
        
        Args:
            model: Model object
            model_name: Name for the model
            framework: ML framework ("sklearn", "keras", etc.)
            signature: Model signature
            input_example: Example input
            registered_model_name: Name for model registry
        """
        try:
            if framework == "sklearn":
                mlflow.sklearn.log_model(
                    model, 
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            elif framework == "keras":
                mlflow.keras.log_model(
                    model, 
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            else:
                # Generic model logging
                mlflow.pyfunc.log_model(
                    model_name,
                    python_model=model,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            
            logger.info(f"Logged {framework} model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error logging model: {e}")
    
    def register_model(self, model_uri: str, model_name: str, 
                      tags: Dict[str, str] = None,
                      description: str = None) -> str:
        """
        Register model in MLflow Model Registry
        
        Args:
            model_uri: URI of the model
            model_name: Name for registered model
            tags: Model tags
            description: Model description
        
        Returns:
            Model version
        """
        try:
            model_version = mlflow.register_model(
                model_uri, 
                model_name,
                tags=tags
            )
            
            # Update model description if provided
            if description:
                self.client.update_registered_model(
                    name=model_name,
                    description=description
                )
            
            logger.info(f"Registered model: {model_name} v{model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None
    
    def transition_model_stage(self, model_name: str, version: str, 
                             stage: str, archive_existing: bool = True):
        """
        Transition model to different stage
        
        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage ("Staging", "Production", "Archived")
            archive_existing: Archive existing models in target stage
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
    
    def get_model_versions(self, model_name: str, 
                          stages: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get model versions from registry
        
        Args:
            model_name: Registered model name
            stages: Filter by stages
        
        Returns:
            List of model version information
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if stages:
                versions = [v for v in versions if v.current_stage in stages]
            
            version_info = []
            for version in versions:
                version_info.append({
                    'version': version.version,
                    'stage': version.current_stage,
                    'creation_timestamp': version.creation_timestamp,
                    'last_updated_timestamp': version.last_updated_timestamp,
                    'run_id': version.run_id,
                    'status': version.status,
                    'tags': dict(version.tags) if version.tags else {}
                })
            
            return version_info
            
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []
    
    def load_model(self, model_name: str, version: str = None, 
                   stage: str = None):
        """
        Load model from registry
        
        Args:
            model_name: Registered model name
            version: Specific version to load
            stage: Stage to load from ("Production", "Staging")
        
        Returns:
            Loaded model
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def search_runs(self, filter_string: str = "", 
                   order_by: List[str] = None,
                   max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        Search runs in the experiment
        
        Args:
            filter_string: Filter criteria
            order_by: Ordering criteria
            max_results: Maximum results to return
        
        Returns:
            List of run information
        """
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results
            )
            
            run_info = []
            for run in runs:
                run_info.append({
                    'run_id': run.info.run_id,
                    'experiment_id': run.info.experiment_id,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'metrics': dict(run.data.metrics),
                    'params': dict(run.data.params),
                    'tags': dict(run.data.tags)
                })
            
            return run_info
            
        except Exception as e:
            logger.error(f"Error searching runs: {e}")
            return []
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get experiment information
        
        Returns:
            Experiment information dictionary
        """
        try:
            experiment = self.client.get_experiment(self.experiment_id)
            
            return {
                'experiment_id': experiment.experiment_id,
                'name': experiment.name,
                'lifecycle_stage': experiment.lifecycle_stage,
                'artifact_location': experiment.artifact_location,
                'creation_time': experiment.creation_time,
                'last_update_time': experiment.last_update_time,
                'tags': dict(experiment.tags) if experiment.tags else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting experiment info: {e}")
            return {}
    
    def cleanup_old_runs(self, max_runs: int = 100):
        """
        Cleanup old runs to manage storage
        
        Args:
            max_runs: Maximum number of runs to keep
        """
        try:
            runs = self.search_runs(order_by=["start_time DESC"])
            
            if len(runs) > max_runs:
                runs_to_delete = runs[max_runs:]
                
                for run in runs_to_delete:
                    self.client.delete_run(run['run_id'])
                    logger.debug(f"Deleted run: {run['run_id']}")
                
                logger.info(f"Deleted {len(runs_to_delete)} old runs")
            
        except Exception as e:
            logger.error(f"Error cleaning up runs: {e}")

# Global MLflow manager instance
mlflow_manager = MLflowManager()

# Convenience functions
def log_system_training_run(model_type: str, metrics: Dict[str, float], 
                           params: Dict[str, Any], model=None):
    """
    Log a complete training run for system performance models
    
    Args:
        model_type: Type of model (LSTM, anomaly_detection, etc.)
        metrics: Performance metrics
        params: Model parameters
        model: Trained model object
    """
    with mlflow_manager.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow_manager.log_params(params)
        
        # Log metrics
        mlflow_manager.log_metrics(metrics)
        
        # Log model if provided
        if model:
            framework = "keras" if hasattr(model, 'fit') and 'tensorflow' in str(type(model)) else "sklearn"
            mlflow_manager.log_model(
                model, 
                model_type,
                framework=framework,
                registered_model_name=f"system_optimizer_{model_type}"
            )
        
        logger.info(f"Logged complete training run for {model_type}")
'''

with open('ml_system_optimizer/mlflow/mlflow_setup.py', 'w') as f:
    f.write(mlflow_setup_content)

print("✅ mlflow_setup.py created successfully!")