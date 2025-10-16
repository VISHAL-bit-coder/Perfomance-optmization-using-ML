# Create training script
run_training_content = '''#!/usr/bin/env python3
"""
Training script for ML System Optimizer
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_trainer import ModelTrainer
from src.config.settings import get_config
from src.config.logging_config import setup_logging
import logging

def main():
    parser = argparse.ArgumentParser(description='Train ML models for system optimization')
    parser.add_argument('--duration', '-d', type=int, default=24,
                       help='Hours of data to collect for training (default: 24)')
    parser.add_argument('--models', '-m', nargs='+', 
                       choices=['forecasting', 'anomaly', 'all'],
                       default=['all'],
                       help='Models to train (default: all)')
    parser.add_argument('--config', '-c', type=str,
                       help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup configuration
    config = get_config()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else config.LOG_LEVEL
    setup_logging(log_level, config.LOG_FILE)
    logger = logging.getLogger('ml_system_optimizer.training')
    
    logger.info("Starting ML System Optimizer training")
    logger.info(f"Training duration: {args.duration} hours")
    logger.info(f"Models to train: {args.models}")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        if 'all' in args.models:
            # Train all models
            logger.info("Training all models...")
            results = trainer.full_training_pipeline(args.duration)
            
            logger.info("Training completed successfully!")
            logger.info(f"Models trained: {list(results['models'].keys())}")
            
            # Print summary
            print("\\n" + "="*50)
            print("TRAINING SUMMARY")
            print("="*50)
            print(f"Data samples: {results['data_info']['samples']}")
            print(f"Features: {results['data_info']['features']}")
            print(f"Time range: {results['data_info']['time_range']['start']} to {results['data_info']['time_range']['end']}")
            print(f"Models trained: {len(results['models'])}")
            
            for model_name, model_info in results['models'].items():
                print(f"\\n{model_name.upper()}:")
                if 'metrics' in model_info:
                    for metric, value in model_info['metrics'].items():
                        print(f"  {metric}: {value:.4f}")
        
        else:
            # Train specific models
            df = trainer.collect_training_data(args.duration)
            
            if 'forecasting' in args.models:
                logger.info("Training forecasting models...")
                for metric in ['cpu_percent', 'memory_percent']:
                    results = trainer.train_forecasting_model(df, metric)
                    logger.info(f"Forecasting model for {metric} trained successfully")
            
            if 'anomaly' in args.models:
                logger.info("Training anomaly detection model...")
                results = trainer.train_anomaly_detection_model(df)
                logger.info("Anomaly detection model trained successfully")
        
        logger.info("All training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''

with open('ml_system_optimizer/scripts/run_training.py', 'w') as f:
    f.write(run_training_content)

print("✅ run_training.py created successfully!")