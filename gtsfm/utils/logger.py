"""Utilities for logging.

Authors: Ayush Baid, John Lambert.
"""

import logging
import socket
import sys
from datetime import datetime, timezone
from logging import Logger

from dask import distributed

# ============================================================================
# Machine-local lookup map (cached worker identity)
# ============================================================================
# This variable is set once per process and never changes.
# It avoids repeated Dask API calls for every log statement.
_WORKER_ID_CACHE: str | None = None


def _detect_worker_id_once() -> str:
    """
    Detect the current process's worker identity exactly once.
    
    This function queries the Dask distributed system to determine if we're
    running inside a Dask worker or in the main scheduler process.
    
    Returns:
        str: Worker ID in format "hostname-w<port>" for workers,
             or "hostname-main" for the main process.
             
    Examples:
        - Dask worker: "hornet-w35163"
        - Main process: "eagle-main"
    """
    try:
        
        # Attempt to get the current worker object.
        # This only succeeds if we're inside a Dask worker process.
        worker = distributed.get_worker()
        
        # Success: we're in a Dask worker
        hostname = socket.gethostname()
        worker_address = worker.address  # Format: "tcp://130.207.121.32:35163"
        
        # Extract port number as a unique identifier for this worker
        port = worker_address.split(":")[-1]
        
        return f"{hostname}-w{port}"
        
    except (ImportError, ValueError, AttributeError):
        # Failure: we're in the main process (or Dask is not available)
        hostname = socket.gethostname()
        return f"{hostname}-main"


class DaskWorkerFilter(logging.Filter):
    """
    Logging filter that injects Dask worker information into log records.
    
    This filter adds a 'worker_id' field to each LogRecord, which can then
    be used in the log format string. The worker_id is detected once and
    cached for the lifetime of the process.
    
    Performance characteristics:
        - First log call: 1 Dask API call (detection)
        - All subsequent calls: 1 variable read (O(1), extremely fast)
    """
    
    def filter(self, record):
        """
        Add worker_id field to the log record (reading from cache).
        
        This method is called for every log statement, but it only reads
        a cached variable - no Dask API calls are made after the first log!
        
        Args:
            record: logging.LogRecord object to be modified
            
        Returns:
            bool: Always True (allow log to pass through)
        """
        global _WORKER_ID_CACHE
        
        # Lazy initialization: detect worker ID on first log call
        if _WORKER_ID_CACHE is None:
            _WORKER_ID_CACHE = _detect_worker_id_once()
        
        # Inject the cached worker_id into the log record
        record.worker_id = _WORKER_ID_CACHE
        
        return True


class UTCFormatter(logging.Formatter):
    """Custom formatter that uses UTC timestamps."""
    
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_logger() -> Logger:
    """
    Get the main logger with automatic Dask worker awareness.
    
    All modules using this logger will automatically get worker information
    in their log output without any code changes!
    
    Log format:
        Before: "2025-10-27 04:01:11 1013142 [filename.py] INFO: message"
        After:  "2025-10-27 04:01:11 [hornet-w35163] [filename.py] INFO: message"
    
    Returns:
        Logger: Configured logger instance
    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Updated format: use %(worker_id)s instead of %(process)d
        fmt = "%(asctime)s [%(worker_id)s] [%(filename)s] %(levelname)s: %(message)s"
        handler.setFormatter(UTCFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        
        # Add the Dask worker filter
        handler.addFilter(DaskWorkerFilter())
        
        logger.addHandler(handler)
        
        # Silence noisy loggers
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.ERROR)
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.ERROR)
    
    return logger