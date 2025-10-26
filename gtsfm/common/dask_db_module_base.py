"""PostgreSQL database client for GTSfM.

This module provides a client class for interacting with PostgreSQL databases,
handling connection management, query execution, and serialization for use
within the GTSfM distributed computing environment.

Authors: Zongyue Liu
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.common.postgres_client import PostgresClient

logger = logger_utils.get_logger()


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
        """
        Custom serialization to avoid serializing the database connection.

        Args:
            None

        Returns:
            Dict[str, Any]: Dictionary containing the object state with database connection removed
        """
        state = self.__dict__.copy()
        # Keep connection parameters but not the connection object
        if "db" in state and state["db"] is not None:
            # Prevent serialization of PostgresClient's connection object
            state["db"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Reinitialize the database client upon deserialization

        Args:
            state: Dictionary containing the object state to restore

        Returns:
            None
        """
        self.__dict__.update(state)
        # Only recreate PostgresClient if postgres_params exists and is not None
        if hasattr(self, "postgres_params") and self.postgres_params:
            try:
                self.db = PostgresClient(self.postgres_params)
                logger.debug("Database client recreated after deserialization")
                # Initialize database tables if needed - each subclass handles its own schema
                self.init_tables()
            except Exception as e:
                logger.error(f"Failed to recreate database client after deserialization: {e}")
                self.db = None
        else:
            # For objects that don't have postgres functionality (like cachers)
            logger.debug("No postgres_params found, setting db to None")
            self.db = None

    def init_tables(self) -> None:
        """
        Override in subclass to initialize required database tables.

        Args:
            None

        Returns:
            None
        """
        pass

    def serialize_data(self, matrix: Optional[Union[np.ndarray, Dict, List, Any]]) -> Optional[str]:
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

    def deserialize_data(self, serialized_data: Optional[str]) -> Any:
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
            # Convert list to numpy array if it contains numeric data
            if isinstance(data, list) and len(data) > 0:
                # Check if it looks like numeric data (handles both 1D and 2D)
                if isinstance(data[0], (list, int, float)):
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

        Returns:
            None
        """
        if success:
            logger.debug(f"Database operation successful: {operation}")
        else:
            logger.error(f"Database operation failed: {operation}. Error: {error_msg}")
