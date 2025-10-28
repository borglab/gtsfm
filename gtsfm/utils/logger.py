"""Utilities for logging.

Authors: Ayush Baid, John Lambert.
"""

import logging
import re
import socket
import sys
from datetime import datetime, timezone
from logging import Logger, LoggerAdapter, LogRecord

# ============================================================================
# Machine-local lookup map (cached worker identity)
# ============================================================================
# This variable is set once per process and never changes.
_WORKER_ID_CACHE: str | None = None


def _detect_worker_id_once() -> str:
    """
    Detect the current process's worker identity with sequential numbering.
    
    Returns:
        str: Worker ID in format "hostname N" where N is sequential worker number,
             or "hostname main" for the main process.
             
    Examples:
        - Dask worker: "eagle 2", "hornet 1", "wildcat 3"
        - Main process: "eagle main"
    """
    try:
        from dask import distributed

        # Attempt to get the current worker object
        worker = distributed.get_worker()
        
        # Success: we're in a Dask worker
        hostname = socket.gethostname().split('.')[0]  # Just the hostname, no domain
        worker_address = worker.address  # Format: "tcp://130.207.121.32:35163"
        my_port = int(worker_address.split(":")[-1])
        
        try:
            # Query the scheduler to get all workers on this machine
            client = distributed.get_client()
            all_workers = client.scheduler_info()['workers']
            
            # Filter workers on the same hostname and sort by port for consistency
            same_host_workers = []
            for worker_addr, worker_info in all_workers.items():
                worker_host = worker_info.get('host', '').split('.')[0]
                if worker_host == hostname:
                    port = int(worker_addr.split(":")[-1])
                    same_host_workers.append(port)
            
            # Sort ports to get consistent numbering
            same_host_workers.sort()
            
            # Find our index (1-based numbering)
            if my_port in same_host_workers:
                worker_num = same_host_workers.index(my_port) + 1
                return f"{hostname} {worker_num}"
            else:
                # Fallback if we can't find ourselves
                return f"{hostname} w{my_port}"
                
        except Exception:
            # Fallback: use simpler approach
            worker_num = my_port % 100
            return f"{hostname} {worker_num}"
        
    except (ImportError, ValueError, AttributeError):
        # Failure: we're in the main process
        hostname = socket.gethostname().split('.')[0]
        return f"{hostname} main"


def get_worker_id() -> str:
    """
    Get the cached worker ID for the current process.
    
    Returns:
        str: Worker ID like "hornet 2" or "eagle main"
    """
    global _WORKER_ID_CACHE
    
    if _WORKER_ID_CACHE is None:
        _WORKER_ID_CACHE = _detect_worker_id_once()
    
    return _WORKER_ID_CACHE


class WorkerAwareAdapter(LoggerAdapter):
    """
    LoggerAdapter that embeds worker ID into the message.
    
    The worker ID is embedded with a special marker that the formatter
    can extract and move to the filename field.
    """
    
    def __init__(self, logger):
        """Initialize with empty extra dict."""
        super().__init__(logger, {})
        # Worker detection happens lazily on first log call
    
    def process(self, msg, kwargs):
        """
        Process the log message by embedding worker ID with a marker.
        
        The marker format is: [WORKER:worker_id]
        This will be extracted by the formatter and placed in the filename field.
        """
        global _WORKER_ID_CACHE
        
        # Lazy detection on first log call in this process
        if _WORKER_ID_CACHE is None:
            _WORKER_ID_CACHE = _detect_worker_id_once()
        
        # Embed worker ID with a special marker that the formatter can extract
        modified_msg = f"[WORKER:{_WORKER_ID_CACHE}]{msg}"
        return modified_msg, kwargs


class DaskAwareFormatter(logging.Formatter):
    """
    Custom formatter that extracts worker ID from message and moves it to filename field.
    
    This creates clean, readable logs with worker identification in the
    filename position: [eagle 2: image_pairs_generator.py]
    """
    
    # Pattern to match [WORKER:worker_id] at the start of the message
    WORKER_PATTERN = re.compile(r'^\[WORKER:(.*?)\]')
    
    def format(self, record: LogRecord) -> str:
        """Format the log record with worker ID extracted and moved to filename field."""
        
        # Extract worker ID from the message
        match = self.WORKER_PATTERN.match(record.getMessage())
        
        if match:
            worker_id = match.group(1)
            # Remove the worker marker from the message
            record.msg = self.WORKER_PATTERN.sub('', record.msg, count=1)
            # If msg was empty after removal, getMessage() might still have it cached
            # Force update by accessing the private cache
            if hasattr(record, '_getMessage'):
                delattr(record, '_getMessage')
        else:
            # No worker marker found (shouldn't happen, but fallback gracefully)
            worker_id = "unknown"
        
        # Inject worker_id into the filename field
        # Original: [image_pairs_generator.py]
        # Modified: [eagle 2: image_pairs_generator.py]
        original_filename = record.filename
        record.filename = f"{worker_id}: {original_filename}"
        
        # Format the message with standard formatter
        formatted = super().format(record)
        
        # Restore original filename to avoid side effects
        record.filename = original_filename
        
        return formatted
    
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
    
    The magic happens through:
    1. LoggerAdapter embeds worker ID in the message (survives Dask forwarding)
    2. Formatter extracts worker ID and moves it to filename field (clean output)
    
    Log format:
        "2025-10-28 00:00:45 [hornet 2: cluster_optimizer.py] INFO: message"
    
    Returns:
        LoggerAdapter: Configured logger adapter instance
    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Worker ID will be extracted from message and injected into %(filename)s
        fmt = "%(asctime)s [%(filename)s] %(levelname)s: %(message)s"
        handler.setFormatter(DaskAwareFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        
        logger.addHandler(handler)
        
        # Silence noisy loggers
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.ERROR)
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.ERROR)
    
    # Return a LoggerAdapter that embeds worker ID in messages
    return WorkerAwareAdapter(logger)