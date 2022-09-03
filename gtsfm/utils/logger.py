"""Utilities for logging.

Authors: Ayush Baid, John Lambert.
"""
import logging
import sys
from logging import Logger


def get_logger() -> Logger:
    """Getter for the main logger."""
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger
