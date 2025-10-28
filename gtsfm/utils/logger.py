"""Utilities for logging.

Authors: Ayush Baid, John Lambert.
"""

import logging
import socket
import sys
from datetime import datetime, timezone
from logging import Logger, LogRecord

# ============================================================================
# Machine-local lookup map (cached worker identity)
# ============================================================================
# This variable is set once per process and never changes.
_WORKER_ID_CACHE: str | None = None


def _detect_worker_id_once() -> str:
    """
    Detect the current process's worker identity with sequential numbering.
    
    This function queries the Dask distributed system to determine worker identity
    and assigns a sequential worker number (1, 2, 3...) for each machine.
    
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
            # This allows us to assign sequential numbers
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
                # Fallback if we can't find ourselves (shouldn't happen)
                return f"{hostname} w{my_port}"
                
        except Exception:
            # Fallback: if we can't query scheduler, use a simpler approach
            # Use last 2 digits of port as worker number
            worker_num = my_port % 100
            return f"{hostname} {worker_num}"
        
    except (ImportError, ValueError, AttributeError):
        # Failure: we're in the main process (or Dask is not available)
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


class DaskAwareFormatter(logging.Formatter):
    """
    Custom formatter that injects worker ID into the filename field.
    
    This creates clean, readable logs with worker identification in the
    filename position: [eagle 2: image_pairs_generator.py]
    """
    
    def format(self, record: LogRecord) -> str:
        """Format the log record with worker ID in filename field."""
        global _WORKER_ID_CACHE
        
        # Lazy detection on first log in this process
        if _WORKER_ID_CACHE is None:
            _WORKER_ID_CACHE = _detect_worker_id_once()
        
        # Inject worker_id into the filename field
        # Original: [image_pairs_generator.py]
        # Modified: [eagle 2: image_pairs_generator.py]
        original_filename = record.filename
        record.filename = f"{_WORKER_ID_CACHE}: {original_filename}"
        
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


def get_logger() -> Logger:
    """
    Get the main logger with automatic Dask worker awareness.
    
    All modules using this logger will automatically get worker information
    in their log output without any code changes!
    
    Worker detection happens lazily at first log call. The worker ID
    appears in the filename field for clean, readable output.
    
    Log format:
        "2025-10-28 00:00:45 [hornet 2: cluster_optimizer.py] INFO: message"
    
    Returns:
        Logger: Configured logger instance
    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Worker ID will be injected into %(filename)s by the formatter
        fmt = "%(asctime)s [%(filename)s] %(levelname)s: %(message)s"
        handler.setFormatter(DaskAwareFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        
        logger.addHandler(handler)
        
        # Silence noisy loggers
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.ERROR)
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.ERROR)
    
    return logger