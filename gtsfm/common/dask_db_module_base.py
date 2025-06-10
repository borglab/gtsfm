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
from typing import Any, Dict, Optional
from gtsfm.common.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


class DaskDBModuleBase:
    """Base class for all modules running on Dask that interact with the database"""
    
    def __init__(self, postgres_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the base class
        
        Args:
            postgres_params: PostgreSQL connection parameters
        """
        self.postgres_params: Optional[Dict[str, Any]] = postgres_params
        self.db: Optional[PostgresClient] = None
        if postgres_params:
            self.db = PostgresClient(postgres_params)
    
    def __getstate__(self) -> Dict[str, Any]:
        """Custom serialization to avoid serializing the database connection"""
        state = self.__dict__.copy()
        # Keep connection parameters but not the connection object
        if 'db' in state and state['db'] is not None:
            # Prevent serialization of PostgresClient's connection object
            state['db'] = None
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Reinitialize the database client upon deserialization"""
        self.__dict__.update(state)
        # Only recreate PostgresClient if postgres_params exists and is not None
        if hasattr(self, 'postgres_params') and self.postgres_params:
            self.db = PostgresClient(self.postgres_params)
            # Initialize database tables if needed
            self.init_database()
        else:
            # For objects that don't have postgres functionality (like cachers)
            self.db = None
    
    def init_database(self) -> None:
        """Override in subclass to initialize required database tables"""
        pass
    
    def serialize_matrix(self, matrix: Any) -> Optional[str]:
        """
        Serialize a NumPy matrix into JSON
        
        Args:
            matrix: NumPy array or object that can be converted to JSON
            
        Returns:
            JSON string or None if matrix is None
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
    
    def deserialize_matrix(self, serialized_data: Optional[str]) -> Any:
        """
        Deserialize a matrix from JSON or pickle
        
        Args:
            serialized_data: Serialized data string
            
        Returns:
            Deserialized object or None if deserialization fails
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
    
    def log_database_operation(self, operation: str, success: bool, error_msg: Optional[str] = None) -> None:
        """
        Log database operations for debugging
        
        Args:
            operation: Description of the operation
            success: Whether the operation was successful
            error_msg: Error message if operation failed
        """
        if success:
            logger.debug(f"Database operation successful: {operation}")
        else:
            logger.error(f"Database operation failed: {operation}. Error: {error_msg}")
