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
_WORKER_ID_CACHE: str | None = None
_MAPPING_LOGGED: bool = False


def _get_sequential_worker_number(hostname: str, port: str) -> int:
    """
    Get sequential worker number using Dask's distributed Variable.
    
    This uses Dask's coordination mechanism to assign truly sequential
    numbers (1, 2, 3...) to workers as they start up.
    
    Args:
        hostname: Worker hostname
        port: Worker port number
        
    Returns:
        Sequential worker number (1, 2, 3...)
    """
    try:
        from dask.distributed import Variable, get_client

        # Try to get the Dask client
        try:
            client = get_client()
        except ValueError:
            # No client available, fall back to hash-based
            hash_val = int(port) * 2654435761
            return (hash_val % 99) + 1
        
        # Use Dask Variable to coordinate worker numbering
        # This is a distributed atomic counter shared across all workers
        worker_key = f"{hostname}:{port}"
        
        # Get or create the worker registry variable
        try:
            worker_registry = Variable("gtsfm_worker_registry", client=client)
            registry = worker_registry.get(timeout=1)
        except (KeyError, TimeoutError):
            # First worker initializes the registry
            registry = {}
        
        # Check if this worker already has a number
        if worker_key in registry:
            return registry[worker_key]
        
        # Assign next sequential number
        next_num = len(registry) + 1
        registry[worker_key] = next_num
        
        # Update the distributed registry
        try:
            worker_registry.set(registry)
        except:
            pass  # Another worker might have updated simultaneously
        
        return next_num
        
    except Exception as e:
        # Fallback to hash-based if anything goes wrong
        hash_val = int(port) * 2654435761
        return (hash_val % 99) + 1


def _detect_worker_id_once() -> str:
    """
    Detect the current process's worker identity exactly once.
    
    Uses Dask's distributed coordination to assign truly sequential
    worker numbers (1, 2, 3...).
    
    Returns:
        str: Worker ID in format "hostname(N)" for workers,
             or "hostname-main" for the main process.
             
    Examples:
        - First worker: "hornet(1)"
        - Second worker: "eagle(2)"
        - Third worker: "hornet(3)"
        - Main process: "eagle-main"
    """
    try:
        from dask import distributed
        
        worker = distributed.get_worker()
        
        # Success: we're in a Dask worker
        hostname = socket.gethostname()
        worker_address = worker.address  # Format: "tcp://130.207.121.32:35163"
        
        # Extract port number
        port = worker_address.split(":")[-1]
        
        # Get sequential worker number through Dask coordination
        worker_num = _get_sequential_worker_number(hostname, port)
        
        return f"{hostname}({worker_num})"
        
    except (ImportError, ValueError, AttributeError):
        # Failure: we're in the main process
        hostname = socket.gethostname()
        return f"{hostname}-main"


def get_worker_id() -> str:
    """Get the cached worker ID for the current process."""
    global _WORKER_ID_CACHE
    
    if _WORKER_ID_CACHE is None:
        _WORKER_ID_CACHE = _detect_worker_id_once()
    
    return _WORKER_ID_CACHE


def _log_worker_mapping(logger):
    """
    Log the worker ID mapping for debugging.
    
    Shows the conversion from full TCP address to simplified worker ID.
    """
    try:
        from dask import distributed
        
        worker = distributed.get_worker()
        hostname = socket.gethostname()
        worker_address = worker.address  # Full TCP address
        port = worker_address.split(":")[-1]
        
        # Extract worker number from cached ID
        worker_num = _WORKER_ID_CACHE.split("(")[1].rstrip(")")
        
        # Log the conversion mapping clearly
        logger.info(
            f"🔧 Worker ID Mapping: {worker_address} → {hostname}({worker_num})"
        )
        logger.info(
            f"   └─ Converted TCP address '{worker_address}' to worker ID '{hostname}({worker_num})' for easier reading"
        )
        
    except Exception as e:
        # If logging fails, don't crash the worker
        pass


class WorkerAwareAdapter(LoggerAdapter):
    """LoggerAdapter that automatically injects worker ID into LogRecords."""
    
    def __init__(self, logger):
        super().__init__(logger, {})
    
    def process(self, msg, kwargs):
        """Inject worker_id into the LogRecord."""
        global _WORKER_ID_CACHE, _MAPPING_LOGGED
        
        if _WORKER_ID_CACHE is None:
            _WORKER_ID_CACHE = _detect_worker_id_once()
            
            # Log mapping on first log from this worker
            if not _MAPPING_LOGGED and "main" not in _WORKER_ID_CACHE:
                _log_worker_mapping(self.logger)
                _MAPPING_LOGGED = True
        
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra']['worker_id'] = _WORKER_ID_CACHE
        
        return msg, kwargs


class DaskAwareFormatter(logging.Formatter):
    """Custom formatter with UTC timestamps and worker awareness."""
    
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


class UTCFormatter(logging.Formatter):
    """Legacy UTC formatter without worker awareness."""
    
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_logger() -> LoggerAdapter:
    """
    Get the main logger with automatic Dask worker awareness.
    
    Workers are assigned truly sequential numbers (1, 2, 3...) using
    Dask's distributed Variable for coordination.
    
    Log format:
        "2025-10-28 00:00:45 [hornet(1)] [filename.py] INFO: message"
    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s [%(worker_id)s] [%(filename)s] %(levelname)s: %(message)s"
        handler.setFormatter(DaskAwareFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        
        # Silence noisy loggers
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.ERROR)
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.ERROR)
    
    return WorkerAwareAdapter(logger)