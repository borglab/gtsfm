"""PostgreSQL database client for GTSFM.

This module provides a client class for interacting with PostgreSQL databases, 
handling connection management, query execution, and serialization for use 
within the GTSFM distributed computing environment.

Authors: Zongyue Liu
"""
import pickle
import numpy as np
import json
import logging
from gtsfm.common.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


class DaskDBModuleBase:
    """Base class for all modules running on Dask that interact with the database"""
    
    def __init__(self, postgres_params=None):
        """
        Initialize the base class
        
        Args:
            postgres_params (dict, optional): PostgreSQL connection parameters
        """
        self.postgres_params = postgres_params
        self.db = None
        if postgres_params:
            self.db = PostgresClient(postgres_params)
    
    def __getstate__(self):
        """Custom serialization to avoid serializing the database connection"""
        state = self.__dict__.copy()
        # Keep connection parameters but not the connection object
        if 'db' in state and state['db'] is not None:
            # Prevent serialization of PostgresClient's connection object
            state['db'] = None
        return state
    
    def __setstate__(self, state):
        """Reinitialize the database client upon deserialization"""
        self.__dict__.update(state)
        # Recreate PostgresClient on the worker
        if self.postgres_params:
            self.db = PostgresClient(self.postgres_params)
            # Initialize database tables if needed
            self.init_database()
    
    def init_database(self):
        """Override in subclass to initialize required database tables"""
    
    def serialize_matrix(self, matrix):
        """
        Serialize a NumPy matrix into JSON
        
        Args:
            matrix: NumPy array or object that can be converted to JSON
            
        Returns:
            str: JSON string
        """
        if matrix is None:
            return None
        
        # Handle NumPy arrays
        if isinstance(matrix, np.ndarray):
            return json.dumps(matrix.tolist())
        
        # Handle dictionaries and lists
        if isinstance(matrix, (dict, list)):
            try:
                return json.dumps(matrix)
            except TypeError:
                # If not JSON serializable, use pickle as a fallback
                return pickle.dumps(matrix).hex()
        
        # Attempt JSON serialization for other types
        try:
            return json.dumps(matrix)
        except (TypeError, ValueError):
            # If not JSON serializable, use pickle as a fallback
            return pickle.dumps(matrix).hex()
    
    def deserialize_matrix(self, serialized_data):
        """
        Deserialize a matrix from JSON or pickle
        
        Args:
            serialized_data (str): Serialized data
            
        Returns:
            object: Deserialized object
        """
        if serialized_data is None:
            return None
        
        try:
            # Try JSON first
            data = json.loads(serialized_data)
            # Convert list back to numpy array if it looks like a matrix
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list):  # 2D array
                    return np.array(data)
                elif isinstance(data[0], (int, float)):  # 1D array
                    return np.array(data)
            return data
        except (json.JSONDecodeError, ValueError):
            # Try pickle if JSON fails
            try:
                return pickle.loads(bytes.fromhex(serialized_data))
            except (ValueError, pickle.UnpicklingError):
                logger.warning(f"Failed to deserialize data: {serialized_data[:50]}...")
                return None
    
    def log_database_operation(self, operation, success, error_msg=None):
        """
        Log database operations for debugging
        
        Args:
            operation (str): Description of the operation
            success (bool): Whether the operation was successful
            error_msg (str, optional): Error message if operation failed
        """
        if success:
            logger.debug(f"Database operation successful: {operation}")
        else:
            logger.error(f"Database operation failed: {operation}. Error: {error_msg}")
