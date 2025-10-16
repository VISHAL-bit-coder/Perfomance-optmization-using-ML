# Create comprehensive system monitoring module
system_monitor_content = '''"""
System monitoring module using psutil to collect system performance metrics
"""
import psutil
import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import threading
import queue
from dataclasses import dataclass
import pandas as pd

from ..config.settings import get_config
from ..config.logging_config import system_logger

@dataclass
class SystemMetrics:
    """Data class for system metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage_percent: float
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int
    boot_time: float
    process_count: int
    load_average_1min: float
    load_average_5min: float
    load_average_15min: float

class SystemMonitor:
    """
    Comprehensive system monitoring class that collects various system metrics
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.monitoring = False
        self.data_queue = queue.Queue()
        self.db_path = self.config.get_paths()['base'] / 'system_metrics.db'
        self._init_database()
        
        # Previous values for calculating rates
        self._prev_disk_io = None
        self._prev_network_io = None
        self._prev_timestamp = None
    
    def _init_database(self):
        """Initialize SQLite database for storing metrics"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_available INTEGER,
                    disk_usage_percent REAL,
                    disk_io_read INTEGER,
                    disk_io_write INTEGER,
                    network_io_sent INTEGER,
                    network_io_recv INTEGER,
                    boot_time REAL,
                    process_count INTEGER,
                    load_average_1min REAL,
                    load_average_5min REAL,
                    load_average_15min REAL
                )
            ''')
            
            # Create index on timestamp for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON system_metrics(timestamp)
            ''')
            
            conn.commit()
            conn.close()
            system_logger.log_system_metrics({"action": "database_initialized", "path": str(self.db_path)})
        except Exception as e:
            system_logger.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def collect_metrics(self) -> SystemMetrics:
        """
        Collect comprehensive system metrics
        
        Returns:
            SystemMetrics object containing all collected metrics
        """
        try:
            current_time = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            disk_io_read = disk_io.read_bytes if disk_io else 0
            disk_io_write = disk_io.write_bytes if disk_io else 0
            
            # Network I/O metrics
            network_io = psutil.net_io_counters()
            network_io_sent = network_io.bytes_sent if network_io else 0
            network_io_recv = network_io.bytes_recv if network_io else 0
            
            # System info
            boot_time = psutil.boot_time()
            process_count = len(psutil.pids())
            
            # Load averages (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                load_average_1min = load_avg[0]
                load_average_5min = load_avg[1]
                load_average_15min = load_avg[2]
            except (AttributeError, OSError):
                # Windows doesn't have load averages
                load_average_1min = 0.0
                load_average_5min = 0.0
                load_average_15min = 0.0
            
            metrics = SystemMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available=memory_available,
                disk_usage_percent=disk_usage_percent,
                disk_io_read=disk_io_read,
                disk_io_write=disk_io_write,
                network_io_sent=network_io_sent,
                network_io_recv=network_io_recv,
                boot_time=boot_time,
                process_count=process_count,
                load_average_1min=load_average_1min,
                load_average_5min=load_average_5min,
                load_average_15min=load_average_15min
            )
            
            system_logger.log_system_metrics(metrics.__dict__)
            return metrics
            
        except Exception as e:
            system_logger.logger.error(f"Error collecting system metrics: {e}")
            raise
    
    def collect_detailed_process_info(self) -> List[Dict]:
        """
        Collect detailed information about running processes
        
        Returns:
            List of dictionaries containing process information
        """
        processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    process_info = proc.info
                    processes.append({
                        'pid': process_info['pid'],
                        'name': process_info['name'],
                        'cpu_percent': process_info['cpu_percent'],
                        'memory_percent': process_info['memory_percent'],
                        'status': process_info['status']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
            return processes[:10]  # Return top 10 processes
            
        except Exception as e:
            system_logger.logger.error(f"Error collecting process information: {e}")
            return []
    
    def save_metrics(self, metrics: SystemMetrics):
        """
        Save metrics to database
        
        Args:
            metrics: SystemMetrics object to save
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_metrics (
                    timestamp, cpu_percent, memory_percent, memory_available,
                    disk_usage_percent, disk_io_read, disk_io_write,
                    network_io_sent, network_io_recv, boot_time, process_count,
                    load_average_1min, load_average_5min, load_average_15min
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.cpu_percent, metrics.memory_percent,
                metrics.memory_available, metrics.disk_usage_percent,
                metrics.disk_io_read, metrics.disk_io_write,
                metrics.network_io_sent, metrics.network_io_recv,
                metrics.boot_time, metrics.process_count,
                metrics.load_average_1min, metrics.load_average_5min,
                metrics.load_average_15min
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            system_logger.logger.error(f"Error saving metrics to database: {e}")
            raise
    
    def get_historical_data(self, hours: int = 24) -> pd.DataFrame:
        """
        Retrieve historical data from database
        
        Args:
            hours: Number of hours of historical data to retrieve
        
        Returns:
            DataFrame containing historical metrics
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            
            start_time = datetime.now() - timedelta(hours=hours)
            
            query = '''
                SELECT * FROM system_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(start_time,))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            conn.close()
            return df
            
        except Exception as e:
            system_logger.logger.error(f"Error retrieving historical data: {e}")
            return pd.DataFrame()
    
    def start_monitoring(self, interval: int = None):
        """
        Start continuous monitoring in a separate thread
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            system_logger.logger.warning("Monitoring is already running")
            return
        
        interval = interval or self.config.MONITORING_INTERVAL
        self.monitoring = True
        
        def monitor_loop():
            system_logger.logger.info(f"Started system monitoring with {interval}s interval")
            
            while self.monitoring:
                try:
                    metrics = self.collect_metrics()
                    self.save_metrics(metrics)
                    
                    # Check for anomalies
                    self._check_thresholds(metrics)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    system_logger.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval)
            
            system_logger.logger.info("System monitoring stopped")
        
        monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        system_logger.logger.info("Stopping system monitoring")
    
    def _check_thresholds(self, metrics: SystemMetrics):
        """
        Check if metrics exceed configured thresholds
        
        Args:
            metrics: SystemMetrics object to check
        """
        # CPU threshold checks
        if metrics.cpu_percent > self.config.CPU_CRITICAL_THRESHOLD:
            system_logger.log_anomaly_detected(
                'cpu_percent', metrics.cpu_percent, self.config.CPU_CRITICAL_THRESHOLD
            )
        elif metrics.cpu_percent > self.config.CPU_WARNING_THRESHOLD:
            system_logger.logger.warning(
                f"CPU usage high: {metrics.cpu_percent:.2f}% (warning threshold: {self.config.CPU_WARNING_THRESHOLD}%)"
            )
        
        # Memory threshold checks
        if metrics.memory_percent > self.config.MEMORY_CRITICAL_THRESHOLD:
            system_logger.log_anomaly_detected(
                'memory_percent', metrics.memory_percent, self.config.MEMORY_CRITICAL_THRESHOLD
            )
        elif metrics.memory_percent > self.config.MEMORY_WARNING_THRESHOLD:
            system_logger.logger.warning(
                f"Memory usage high: {metrics.memory_percent:.2f}% (warning threshold: {self.config.MEMORY_WARNING_THRESHOLD}%)"
            )
        
        # Disk threshold checks
        if metrics.disk_usage_percent > self.config.DISK_CRITICAL_THRESHOLD:
            system_logger.log_anomaly_detected(
                'disk_usage_percent', metrics.disk_usage_percent, self.config.DISK_CRITICAL_THRESHOLD
            )
        elif metrics.disk_usage_percent > self.config.DISK_WARNING_THRESHOLD:
            system_logger.logger.warning(
                f"Disk usage high: {metrics.disk_usage_percent:.2f}% (warning threshold: {self.config.DISK_WARNING_THRESHOLD}%)"
            )
    
    def get_system_summary(self) -> Dict:
        """
        Get current system summary
        
        Returns:
            Dictionary containing system summary
        """
        try:
            metrics = self.collect_metrics()
            processes = self.collect_detailed_process_info()
            
            summary = {
                'timestamp': metrics.timestamp.isoformat(),
                'system': {
                    'cpu_percent': metrics.cpu_percent,
                    'memory_percent': metrics.memory_percent,
                    'memory_available_gb': metrics.memory_available / (1024**3),
                    'disk_usage_percent': metrics.disk_usage_percent,
                    'process_count': metrics.process_count,
                    'uptime_hours': (time.time() - metrics.boot_time) / 3600
                },
                'load_averages': {
                    '1min': metrics.load_average_1min,
                    '5min': metrics.load_average_5min,
                    '15min': metrics.load_average_15min
                },
                'top_processes': processes
            }
            
            return summary
            
        except Exception as e:
            system_logger.logger.error(f"Error generating system summary: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30):
        """
        Clean up old data from database
        
        Args:
            days: Number of days of data to keep
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                DELETE FROM system_metrics WHERE timestamp < ?
            ''', (cutoff_date,))
            
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()
            
            system_logger.logger.info(f"Cleaned up {deleted_rows} old records from database")
            
        except Exception as e:
            system_logger.logger.error(f"Error cleaning up old data: {e}")
            raise

# Global system monitor instance
system_monitor = SystemMonitor()
'''

with open('ml_system_optimizer/src/data_collection/system_monitor.py', 'w') as f:
    f.write(system_monitor_content)

print("✅ system_monitor.py created successfully!")