"""Utilities for logging.

Authors: Ayush Baid, John Lambert.
"""
import logging
import sys
from logging import Logger
import logging.handlers

def get_logger() -> Logger:
    """Getter for the main logger."""
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.handlers.SocketHandler('eagle.cc.gatech.edu', 5000)
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger
