"""Utilities for logging.

Authors: Ayush Baid, John Lambert.
"""

import logging
import socket
import sys
from datetime import datetime, timezone
from logging import Logger

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
        from dask import distributed

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


def get_worker_id() -> str:
    """
    Get the cached worker ID for the current process.
    
    This is thread-safe and only performs the detection once per process.
    
    Returns:
        str: Worker ID like "hornet-w35163" or "eagle-main"
    """
    global _WORKER_ID_CACHE
    
    if _WORKER_ID_CACHE is None:
        _WORKER_ID_CACHE = _detect_worker_id_once()
    
    return _WORKER_ID_CACHE


class DaskAwareFormatter(logging.Formatter):
    """
    Custom formatter that injects worker ID into the formatted message.
    
    This approach works even when logs are collected from workers by Dask,
    because we modify the formatted string, not the LogRecord attributes.
    """
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        # Cache worker_id at formatter creation time
        self._worker_id = get_worker_id()
    
    def format(self, record):
        """Format the log record with worker ID prepended."""
        # Get the standard formatted message
        original_formatted = super().format(record)
        
        # Inject worker ID at the beginning (after timestamp)
        # Original format: "2025-10-28 00:00:45 [filename.py] INFO: message"
        # We want: "2025-10-28 00:00:45 [worker-id] [filename.py] INFO: message"
        
        # Find where to insert (after the timestamp)
        # The timestamp format is: "YYYY-MM-DD HH:MM:SS"
        if len(original_formatted) > 19:  # Ensure timestamp exists
            timestamp_end = 19
            # Insert worker_id after timestamp
            formatted_with_worker = (
                original_formatted[:timestamp_end] + 
                f" [{self._worker_id}]" +
                original_formatted[timestamp_end:]
            )
            return formatted_with_worker
        
        # Fallback if format is unexpected
        return f"[{self._worker_id}] {original_formatted}"
    
    def formatTime(self, record, datefmt=None):
        """Use UTC timestamps."""
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
        "2025-10-28 00:00:45 [hornet-w35163] [filename.py] INFO: message"
    
    Returns:
        Logger: Configured logger instance
    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Simple format without worker_id placeholder (we'll inject it in formatter)
        fmt = "%(asctime)s [%(filename)s] %(levelname)s: %(message)s"
        handler.setFormatter(DaskAwareFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        
        logger.addHandler(handler)
        
        # Silence noisy loggers
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.ERROR)
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.ERROR)
    
    return logger