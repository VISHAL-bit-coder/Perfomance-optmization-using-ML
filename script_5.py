# Create comprehensive system monitoring module (fixed indentation)
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
            
            cursor.execute("""
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
            """)
            
            # Create index on timestamp for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON system_metrics(timestamp)
            """)
            
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
            
            query = """
                SELECT * FROM system_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(start_time,))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            conn.close()
            return df
            
        except Exception as e:
            system_logger.logger.error(f"Error retrieving historical data: {e}")
            return pd.DataFrame()

# Global system monitor instance
system_monitor = SystemMonitor()
'''

with open('ml_system_optimizer/src/data_collection/system_monitor.py', 'w') as f:
    f.write(system_monitor_content)

print("✅ system_monitor.py created successfully!")