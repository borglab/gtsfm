# """Utilities for logging.

# Authors: Ayush Baid, John Lambert.
# """
# import logging
# from logging import Logger
# import logging.handlers
# import socket

# def get_logger() -> Logger:
#     """Getter for the main logger."""
#     logger_name = "main-logger"
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(logging.INFO)

#     logging.basicConfig(
#             format="[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s")

#     if not logger.handlers:
#         handler = logging.handlers.SocketHandler('eagle.cc.gatech.edu', 5000)
#         fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
#         handler.setFormatter(logging.Formatter(fmt))
#         logger.addHandler(handler)
#     return logger


"""Utilities for logging.

Authors: Ayush Baid, John Lambert.
"""
import logging
from logging import Logger
import struct
import pickle
import socketserver

def get_logger() -> Logger:
    """Getter for the main logger."""
    # logging.basicConfig(
    #         format="[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s")

    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        socket_handler = logging.handlers.SocketHandler('eagle.cc.gatech.edu', 5000)
        socket_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        socket_handler.setFormatter(formatter)

        logger.addHandler(socket_handler)

    print(logger, logger.handlers[0])
    return logger


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    def handle(self):
        while True:
            try:
                # Read a log record as a byte string
                chunk = self.connection.recv(4)
                if len(chunk) < 4:
                    break
                slen = struct.unpack(">L", chunk)[0]
                chunk = self.connection.recv(slen)
                while len(chunk) < slen:
                    chunk = chunk + self.connection.recv(slen - len(chunk))
                obj = pickle.loads(chunk)
                record = logging.makeLogRecord(obj)
                self.handle_log_record(record)
            except Exception as e:
                print(e)

    def handle_log_record(self, record):
        logger = logging.getLogger(record.name)
        print(record)
        logger.handle(record)
