import os
import sys
import datetime
from pathlib import Path
import functools

class Logger:
    def __init__(self, log_dir='logs', log_prefix='experiment', enable_file=True, enable_console=True):
        self.enable_file = enable_file
        self.enable_console = enable_console
        
        if self.enable_file:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"{log_prefix}_{timestamp}.log"
            self.log_path = self.log_dir / log_filename
            
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            
            self.log_file = open(self.log_path, 'w', encoding='utf-8', buffering=1)
            
            print(f"log start, save to: {self.log_path}")
    
    def write(self, message):
        if self.enable_console:
            self.original_stdout.write(message)
        if self.enable_file and hasattr(self, 'log_file'):
            self.log_file.write(message)
    
    def flush(self):
        if self.enable_console:
            self.original_stdout.flush()
        if self.enable_file and hasattr(self, 'log_file'):
            self.log_file.flush()
    
    def log(self, *args, **kwargs):
        message_parts = []
        for arg in args:
            message_parts.append(str(arg))
        
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        
        message = sep.join(message_parts) + end
        
        if kwargs.get('timestamp', False):
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = f"[{timestamp}] {message}"
        
        self.write(message)
        self.flush()
    
    def redirect_print(self):
        if self.enable_file:
            sys.stdout = self
    
    def restore_print(self):
        if self.enable_file and hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout
    
    def close(self):
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
        self.restore_print()
    
    def __enter__(self):
        self.redirect_print()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def setup_logging(log_dir='logs', log_prefix='experiment', auto_redirect=True):
    logger = Logger(log_dir=log_dir, log_prefix=log_prefix)
    
    if auto_redirect:
        logger.redirect_print()
    
    return logger

def log_function_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"[{timestamp}] ❌ function {func_name} failed: {e}")
            raise
    
    return wrapper

_global_logger = None

def init_global_logger(log_dir='logs', log_prefix='experiment'):
    global _global_logger
    _global_logger = setup_logging(log_dir=log_dir, log_prefix=log_prefix)
    return _global_logger

def get_global_logger():
    return _global_logger

def cleanup_global_logger():
    global _global_logger
    if _global_logger:
        _global_logger.close()
        _global_logger = None