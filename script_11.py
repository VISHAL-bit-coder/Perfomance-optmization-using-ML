# Create API utility functions
api_utils_content = '''"""
Utility functions for Flask API
"""
from flask import jsonify, request
from functools import wraps
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from ..config.logging_config import api_logger

def validate_request_data(required_fields: List[str] = None, 
                         optional_fields: List[str] = None) -> callable:
    """
    Decorator to validate JSON request data
    
    Args:
        required_fields: List of required fields
        optional_fields: List of optional fields
    
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                return format_response({'error': 'Content-Type must be application/json'}, 400)
            
            data = request.get_json()
            if not data:
                return format_response({'error': 'No JSON data provided'}, 400)
            
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return format_response({
                        'error': f'Missing required fields: {missing_fields}'
                    }, 400)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def format_response(data: Dict[str, Any], status_code: int = 200) -> tuple:
    """
    Format API response consistently
    
    Args:
        data: Response data
        status_code: HTTP status code
    
    Returns:
        Formatted JSON response
    """
    response_data = {
        'timestamp': datetime.now().isoformat(),
        'status': 'success' if status_code < 400 else 'error'
    }
    response_data.update(data)
    
    return jsonify(response_data), status_code

def handle_errors(func):
    """
    Decorator to handle common API errors
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Log successful request
            api_logger.log_request(
                request.method, 
                request.path, 
                result[1] if isinstance(result, tuple) else 200,
                end_time - start_time
            )
            
            return result
            
        except ValueError as e:
            api_logger.log_error(request.path, f"ValueError: {str(e)}")
            return format_response({'error': str(e)}, 400)
        
        except KeyError as e:
            api_logger.log_error(request.path, f"KeyError: {str(e)}")
            return format_response({'error': f'Missing key: {str(e)}'}, 400)
        
        except Exception as e:
            api_logger.log_error(request.path, f"Unexpected error: {str(e)}")
            return format_response({'error': 'Internal server error'}, 500)
    
    return wrapper

def paginate_data(data: List[Dict], page: int = 1, per_page: int = 100) -> Dict[str, Any]:
    """
    Paginate list data
    
    Args:
        data: List of data items
        page: Page number (1-based)
        per_page: Items per page
    
    Returns:
        Paginated data with metadata
    """
    total_items = len(data)
    total_pages = (total_items + per_page - 1) // per_page
    
    if page < 1:
        page = 1
    if page > total_pages and total_pages > 0:
        page = total_pages
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    paginated_data = data[start_idx:end_idx]
    
    return {
        'data': paginated_data,
        'pagination': {
            'current_page': page,
            'per_page': per_page,
            'total_pages': total_pages,
            'total_items': total_items,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
    }

def serialize_numpy(obj):
    """
    Serialize numpy objects for JSON
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    return obj

def validate_time_range(start_time: str = None, end_time: str = None) -> tuple:
    """
    Validate and parse time range parameters
    
    Args:
        start_time: Start time string (ISO format)
        end_time: End time string (ISO format)
    
    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    try:
        if start_time:
            start_dt = pd.to_datetime(start_time)
        else:
            start_dt = datetime.now() - pd.Timedelta(hours=24)
        
        if end_time:
            end_dt = pd.to_datetime(end_time)
        else:
            end_dt = datetime.now()
        
        if start_dt >= end_dt:
            raise ValueError("Start time must be before end time")
        
        # Limit time range to prevent excessive data
        max_range = pd.Timedelta(days=7)
        if end_dt - start_dt > max_range:
            raise ValueError("Time range cannot exceed 7 days")
        
        return start_dt.to_pydatetime(), end_dt.to_pydatetime()
        
    except Exception as e:
        raise ValueError(f"Invalid time format: {str(e)}")

def validate_numeric_params(**params) -> Dict[str, float]:
    """
    Validate numeric parameters
    
    Args:
        **params: Parameter name-value pairs with validation rules
    
    Returns:
        Validated parameters
    """
    validated = {}
    
    for param_name, config in params.items():
        value = config.get('value')
        min_val = config.get('min')
        max_val = config.get('max')
        default = config.get('default')
        
        if value is None:
            if default is not None:
                validated[param_name] = default
            continue
        
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Parameter '{param_name}' must be numeric")
        
        if min_val is not None and value < min_val:
            raise ValueError(f"Parameter '{param_name}' must be >= {min_val}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"Parameter '{param_name}' must be <= {max_val}")
        
        validated[param_name] = value
    
    return validated

def calculate_summary_stats(data: List[float]) -> Dict[str, float]:
    """
    Calculate summary statistics for numeric data
    
    Args:
        data: List of numeric values
    
    Returns:
        Dictionary of summary statistics
    """
    if not data:
        return {}
    
    data_array = np.array(data)
    
    return {
        'count': len(data),
        'mean': float(np.mean(data_array)),
        'median': float(np.median(data_array)),
        'std': float(np.std(data_array)),
        'min': float(np.min(data_array)),
        'max': float(np.max(data_array)),
        'q25': float(np.percentile(data_array, 25)),
        'q75': float(np.percentile(data_array, 75))
    }

def format_system_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format system metrics for API response
    
    Args:
        metrics_dict: Raw metrics dictionary
    
    Returns:
        Formatted metrics dictionary
    """
    formatted = {}
    
    for key, value in metrics_dict.items():
        if key == 'timestamp':
            formatted[key] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
        elif isinstance(value, (int, float)):
            if key.endswith('_percent'):
                formatted[key] = round(float(value), 2)
            elif key.endswith('_bytes') or key.endswith('_available'):
                # Convert bytes to MB/GB for readability
                if value > 1024**3:
                    formatted[f"{key}_gb"] = round(value / (1024**3), 2)
                elif value > 1024**2:
                    formatted[f"{key}_mb"] = round(value / (1024**2), 2)
                else:
                    formatted[key] = int(value)
            else:
                formatted[key] = round(float(value), 3) if isinstance(value, float) else int(value)
        else:
            formatted[key] = serialize_numpy(value)
    
    return formatted

def create_health_status(metrics: Dict[str, Any], thresholds: Dict[str, float]) -> Dict[str, Any]:
    """
    Create health status based on system metrics
    
    Args:
        metrics: System metrics
        thresholds: Warning/critical thresholds
    
    Returns:
        Health status information
    """
    status = {
        'overall': 'healthy',
        'components': {},
        'alerts': []
    }
    
    # Check CPU
    cpu_percent = metrics.get('cpu_percent', 0)
    if cpu_percent > thresholds.get('cpu_critical', 90):
        status['components']['cpu'] = 'critical'
        status['alerts'].append({
            'type': 'cpu',
            'level': 'critical',
            'message': f'CPU usage critical: {cpu_percent:.1f}%'
        })
        status['overall'] = 'critical'
    elif cpu_percent > thresholds.get('cpu_warning', 80):
        status['components']['cpu'] = 'warning'
        status['alerts'].append({
            'type': 'cpu',
            'level': 'warning',
            'message': f'CPU usage high: {cpu_percent:.1f}%'
        })
        if status['overall'] == 'healthy':
            status['overall'] = 'warning'
    else:
        status['components']['cpu'] = 'healthy'
    
    # Check Memory
    memory_percent = metrics.get('memory_percent', 0)
    if memory_percent > thresholds.get('memory_critical', 90):
        status['components']['memory'] = 'critical'
        status['alerts'].append({
            'type': 'memory',
            'level': 'critical',
            'message': f'Memory usage critical: {memory_percent:.1f}%'
        })
        status['overall'] = 'critical'
    elif memory_percent > thresholds.get('memory_warning', 80):
        status['components']['memory'] = 'warning'
        status['alerts'].append({
            'type': 'memory',
            'level': 'warning',
            'message': f'Memory usage high: {memory_percent:.1f}%'
        })
        if status['overall'] == 'healthy':
            status['overall'] = 'warning'
    else:
        status['components']['memory'] = 'healthy'
    
    # Check Disk
    disk_percent = metrics.get('disk_usage_percent', 0)
    if disk_percent > thresholds.get('disk_critical', 90):
        status['components']['disk'] = 'critical'
        status['alerts'].append({
            'type': 'disk',
            'level': 'critical',
            'message': f'Disk usage critical: {disk_percent:.1f}%'
        })
        status['overall'] = 'critical'
    elif disk_percent > thresholds.get('disk_warning', 80):
        status['components']['disk'] = 'warning'
        status['alerts'].append({
            'type': 'disk',
            'level': 'warning',
            'message': f'Disk usage high: {disk_percent:.1f}%'
        })
        if status['overall'] == 'healthy':
            status['overall'] = 'warning'
    else:
        status['components']['disk'] = 'healthy'
    
    return status

class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client
        
        Args:
            client_id: Client identifier
        
        Returns:
            True if request is allowed
        """
        now = time.time()
        
        # Clean old entries
        self.requests = {
            cid: [req_time for req_time in req_times if now - req_time < self.window_seconds]
            for cid, req_times in self.requests.items()
        }
        
        # Check current client
        client_requests = self.requests.get(client_id, [])
        
        if len(client_requests) >= self.max_requests:
            return False
        
        # Add current request
        if client_id not in self.requests:
            self.requests[client_id] = []
        self.requests[client_id].append(now)
        
        return True

# Global rate limiter instance
rate_limiter = RateLimiter()

def apply_rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """
    Rate limiting decorator
    
    Args:
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
    
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            client_id = request.remote_addr or 'unknown'
            
            if not rate_limiter.is_allowed(client_id):
                return format_response({
                    'error': 'Rate limit exceeded',
                    'limit': max_requests,
                    'window_seconds': window_seconds
                }, 429)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
'''

with open('ml_system_optimizer/src/api/utils.py', 'w') as f:
    f.write(api_utils_content)

print("✅ utils.py created successfully!")