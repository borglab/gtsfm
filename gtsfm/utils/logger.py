"""Utilities for logging.

Authors: Ayush Baid, John Lambert.
"""
import logging
from logging import Logger
import logging.handlers
import socket

def get_logger() -> Logger:
    """Getter for the main logger."""
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    logging.basicConfig(
            format=f"[%(asctime)s %(levelname)s {socket.gethostname()} %(filename)s line %(lineno)d %(process)d] %(message)s")

    if not logger.handlers:
        handler = logging.handlers.SocketHandler('eagle.cc.gatech.edu', 5000)
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger
