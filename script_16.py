# Create system monitoring script
monitor_system_content = '''#!/usr/bin/env python3
"""
System monitoring script for ML System Optimizer
"""
import argparse
import sys
import time
import signal
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.system_monitor import SystemMonitor
from src.models.model_trainer import ModelTrainer
from src.config.settings import get_config
from src.config.logging_config import setup_logging
import logging

class SystemMonitorService:
    """Service for continuous system monitoring and optimization"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.monitor = SystemMonitor(self.config)
        self.trainer = ModelTrainer(self.config)
        self.running = False
        self.logger = logging.getLogger('ml_system_optimizer.monitor_service')
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def start(self, interval: int = None, auto_retrain: bool = True):
        """
        Start monitoring service
        
        Args:
            interval: Monitoring interval in seconds
            auto_retrain: Enable automatic model retraining
        """
        interval = interval or self.config.MONITORING_INTERVAL
        
        self.logger.info("Starting system monitoring service")
        self.logger.info(f"Monitoring interval: {interval} seconds")
        self.logger.info(f"Auto-retrain: {auto_retrain}")
        
        self.running = True
        
        # Start system monitoring
        self.monitor.start_monitoring(interval)
        
        # Main service loop
        retrain_counter = 0
        retrain_interval = self.config.MODEL_RETRAIN_INTERVAL // interval
        
        try:
            while self.running:
                time.sleep(interval)
                retrain_counter += 1
                
                # Check for retraining
                if auto_retrain and retrain_counter >= retrain_interval:
                    self.logger.info("Checking if models need retraining...")
                    
                    try:
                        retrained = self.trainer.retrain_if_needed()
                        if retrained:
                            self.logger.info("Models were retrained")
                        else:
                            self.logger.info("Models are up to date")
                    except Exception as e:
                        self.logger.error(f"Error during retraining check: {e}")
                    
                    retrain_counter = 0
                
                # Cleanup old data periodically
                if retrain_counter % 100 == 0:  # Every ~100 intervals
                    try:
                        self.monitor.cleanup_old_data(days=30)
                        self.logger.debug("Cleaned up old monitoring data")
                    except Exception as e:
                        self.logger.error(f"Error cleaning up data: {e}")
        
        except KeyboardInterrupt:
            self.logger.info("Monitoring interrupted by user")
        except Exception as e:
            self.logger.error(f"Monitoring service error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop monitoring service"""
        if self.running:
            self.logger.info("Stopping system monitoring service")
            self.running = False
            self.monitor.stop_monitoring()
    
    def get_status(self) -> dict:
        """Get current service status"""
        return {
            'running': self.running,
            'monitoring_active': self.monitor.monitoring,
            'config': {
                'monitoring_interval': self.config.MONITORING_INTERVAL,
                'retrain_interval': self.config.MODEL_RETRAIN_INTERVAL
            }
        }

def main():
    parser = argparse.ArgumentParser(description='System monitoring service for ML System Optimizer')
    parser.add_argument('--interval', '-i', type=int, default=None,
                       help='Monitoring interval in seconds')
    parser.add_argument('--no-retrain', action='store_true',
                       help='Disable automatic model retraining')
    parser.add_argument('--duration', '-d', type=int, default=None,
                       help='Run for specified duration in seconds (default: run indefinitely)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Run in test mode (collect data for 1 minute)')
    
    args = parser.parse_args()
    
    # Setup configuration
    config = get_config()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else config.LOG_LEVEL
    setup_logging(log_level, config.LOG_FILE)
    logger = logging.getLogger('ml_system_optimizer.monitor')
    
    logger.info("Starting ML System Optimizer monitoring service")
    
    try:
        # Initialize service
        service = SystemMonitorService(config)
        
        if args.test:
            # Test mode - run for 1 minute
            logger.info("Running in test mode for 60 seconds")
            
            # Start monitoring
            service.monitor.start_monitoring(5)  # 5-second interval for testing
            
            # Collect some data
            time.sleep(60)
            
            # Get summary
            summary = service.monitor.get_system_summary()
            print("\\n" + "="*50)
            print("TEST MODE SUMMARY")
            print("="*50)
            print(f"Current CPU: {summary['system']['cpu_percent']:.1f}%")
            print(f"Current Memory: {summary['system']['memory_percent']:.1f}%")
            print(f"Current Disk: {summary['system']['disk_usage_percent']:.1f}%")
            print(f"Process Count: {summary['system']['process_count']}")
            print(f"Uptime: {summary['system']['uptime_hours']:.1f} hours")
            
            # Get historical data
            df = service.monitor.get_historical_data(hours=1)
            print(f"\\nCollected {len(df)} data points")
            
            service.stop()
            logger.info("Test completed successfully")
        
        else:
            # Normal operation
            interval = args.interval or config.MONITORING_INTERVAL
            auto_retrain = not args.no_retrain
            
            if args.duration:
                logger.info(f"Running for {args.duration} seconds")
                
                # Start service in background
                import threading
                service_thread = threading.Thread(
                    target=service.start,
                    args=(interval, auto_retrain)
                )
                service_thread.start()
                
                # Wait for specified duration
                time.sleep(args.duration)
                
                # Stop service
                service.stop()
                service_thread.join()
                
            else:
                # Run indefinitely
                logger.info("Running indefinitely (Ctrl+C to stop)")
                service.start(interval, auto_retrain)
        
        logger.info("Monitoring service stopped successfully")
        
    except Exception as e:
        logger.error(f"Monitoring service failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''

with open('ml_system_optimizer/scripts/monitor_system.py', 'w') as f:
    f.write(monitor_system_content)

print("✅ monitor_system.py created successfully!")