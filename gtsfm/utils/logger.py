"""Utilities for logging.

Authors: Ayush Baid, John Lambert.
"""

import logging
import socket
import sys
from datetime import datetime, timezone
from logging import Logger, LoggerAdapter

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
    
    This function should be called by worker code that wants to include
    worker information in log messages, especially when those logs will
    be collected and displayed in the main process.
    
    Returns:
        str: Worker ID like "hornet-w35163" or "eagle-main"
    """
    global _WORKER_ID_CACHE
    
    if _WORKER_ID_CACHE is None:
        _WORKER_ID_CACHE = _detect_worker_id_once()
    
    return _WORKER_ID_CACHE


class WorkerAwareAdapter(LoggerAdapter):
    """
    LoggerAdapter that automatically injects worker ID into LogRecords.
    
    This is the ELEGANT solution for Dask-aware logging!
    The worker_id is added as an attribute to the LogRecord, which
    the formatter uses. Since formatting happens on the worker side
    before log forwarding, the worker_id appears in the right place.
    
    CRITICAL: Worker detection happens LAZILY at first log call, not at
    adapter creation, because Dask worker context isn't available at import time!
    """
    
    def __init__(self, logger):
        """Initialize with empty extra dict."""
        super().__init__(logger, {})
        # Do NOT detect worker_id here! It's too early.
        # Detection happens in process() method on first log call.
    
    def process(self, msg, kwargs):
        """
        Process the log call by injecting worker_id into the LogRecord.
        
        This runs BEFORE the LogRecord is created. We inject worker_id
        as an 'extra' field so the formatter can access it via %(worker_id)s.
        
        CRITICAL: Worker detection happens HERE (lazily) because at module
        import time the Dask worker context isn't ready yet!
        """
        global _WORKER_ID_CACHE
        
        # Lazy detection on first log call in this process
        if _WORKER_ID_CACHE is None:
            _WORKER_ID_CACHE = _detect_worker_id_once()
        
        # Inject worker_id into the LogRecord's extra fields
        # This allows the formatter to use %(worker_id)s in the format string
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra']['worker_id'] = _WORKER_ID_CACHE
        
        return msg, kwargs


class DaskAwareFormatter(logging.Formatter):
    """
    Custom formatter with UTC timestamps and worker awareness.
    
    The worker_id is injected by WorkerAwareAdapter and accessed
    via %(worker_id)s in the format string.
    """
    
    def formatTime(self, record, datefmt=None):
        """Use UTC timestamps."""
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


class UTCFormatter(logging.Formatter):
    """Legacy UTC formatter without worker awareness (for compatibility)."""
    
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_logger() -> LoggerAdapter:
    """
    Get the main logger with automatic Dask worker awareness.
    
    All modules using this logger will automatically get worker information
    in their log output without any code changes!
    
    The magic happens through a LoggerAdapter that injects the worker ID
    into the LogRecord, and the formatter displays it before the filename.
    Worker detection is LAZY - it happens on the first log call, not at
    logger creation time, because Dask worker context isn't available at
    module import time.
    
    Log format:
        "2025-10-28 00:00:45 [hornet-w35163] [filename.py] INFO: message"
    
    Returns:
        LoggerAdapter: Configured logger adapter instance
    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Format: timestamp [worker-id] [filename] LEVEL: message
        # The %(worker_id)s is injected by WorkerAwareAdapter
        fmt = "%(asctime)s [%(worker_id)s] [%(filename)s] %(levelname)s: %(message)s"
        handler.setFormatter(DaskAwareFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        
        logger.addHandler(handler)
        
        # Silence noisy loggers
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.ERROR)
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.ERROR)
    
    # Return a LoggerAdapter that wraps the logger
    # This adapter will automatically inject worker_id into all log records
    return WorkerAwareAdapter(logger)