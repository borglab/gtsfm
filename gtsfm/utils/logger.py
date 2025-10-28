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
    LoggerAdapter that automatically prepends worker ID to all log messages.
    
    This is the ELEGANT solution for Dask-aware logging!
    LoggerAdapter modifies the message BEFORE creating the LogRecord,
    so the worker_id becomes part of the message text itself and survives
    Dask's log forwarding from workers to the main scheduler.
    
    Unlike Filter (which adds attributes to LogRecord), this approach
    works perfectly with Dask because the modified message is part of
    the standard 'msg' field which is always preserved.
    """
    
    def __init__(self, logger):
        """Initialize with empty extra dict - we'll add worker_id dynamically."""
        super().__init__(logger, {})
        # Detect worker ID once when adapter is created
        global _WORKER_ID_CACHE
        if _WORKER_ID_CACHE is None:
            _WORKER_ID_CACHE = _detect_worker_id_once()
        self.worker_id = _WORKER_ID_CACHE
    
    def process(self, msg, kwargs):
        """
        Process the log message by prepending worker ID.
        
        This runs BEFORE the LogRecord is created, so the worker_id
        becomes part of the message itself and will survive serialization.
        """
        # Prepend worker_id to the message
        # Note: we're NOT modifying the format, just the message content
        modified_msg = f"[{self.worker_id}] {msg}"
        return modified_msg, kwargs


class DaskAwareFormatter(logging.Formatter):
    """
    Custom formatter with UTC timestamps.
    
    Note: With the LoggerAdapter approach, we don't need to do anything
    special here - the worker ID is already in the message!
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
    
    The magic happens through a LoggerAdapter that prepends the worker ID
    to every message BEFORE the LogRecord is created. This means the worker_id
    becomes part of the message text and survives Dask's log forwarding.
    
    Log format:
        "2025-10-28 00:00:45 [filename.py] INFO: [hornet-w35163] message"
    
    Returns:
        LoggerAdapter: Configured logger adapter instance
    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Simple format - worker_id will be in the message itself
        fmt = "%(asctime)s [%(filename)s] %(levelname)s: %(message)s"
        handler.setFormatter(DaskAwareFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        
        logger.addHandler(handler)
        
        # Silence noisy loggers
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.ERROR)
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.ERROR)
    
    # Return a LoggerAdapter that wraps the logger
    # This adapter will automatically prepend worker_id to all messages
    return WorkerAwareAdapter(logger)