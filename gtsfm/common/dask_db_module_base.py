import psycopg2
from datetime import datetime
import pickle
import numpy as np
import json

class PostgresClient:
    """PostgreSQL database client for handling connections and operations"""
    
    def __init__(self, db_params):
        """
        Initialize the PostgreSQL client
        
        Args:
            db_params (dict): Database connection parameters including host, port, database, user, and password
        """
        self.db_params = db_params
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish a database connection"""
        try:
            if self.conn is None or self.conn.closed:
                self.conn = psycopg2.connect(**self.db_params)
                self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    def close(self):
        """Close the database connection"""
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def execute(self, query, params=None):
        """
        Execute an SQL query
        
        Args:
            query (str): SQL query string
            params (tuple, optional): Query parameters
            
        Returns:
            bool: Whether the execution was successful
        """
        # Ensure the connection is created for each operation rather than being held persistently
        if not self.connect():
            return False
        
        try:
            self.cursor.execute(query, params)
            self.conn.commit()
            return True
        except Exception as e:
            print(f"SQL execution failed: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            # Close the connection after execution to avoid open connections during serialization
            self.close()
    
    def fetch_all(self, query, params=None):
        """
        Execute a query and fetch all results
        
        Args:
            query (str): SQL query string
            params (tuple, optional): Query parameters
            
        Returns:
            list: List of query results, or None if failed
        """
        # Ensure the connection is created for each operation rather than being held persistently
        if not self.connect():
            return None
        
        try:
            self.cursor.execute(query, params)
            result = self.cursor.fetchall()
            return result
        except Exception as e:
            print(f"Query execution failed: {e}")
            return None
        finally:
            # Close the connection after execution to avoid open connections during serialization
            self.close()
    
    def __getstate__(self):
        """Custom serialization to avoid serializing connection objects"""
        state = self.__dict__.copy()
        # Do not serialize connection and cursor
        state['conn'] = None
        state['cursor'] = None
        return state


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
    
    def init_database(self):
        """Override in subclass to initialize required database tables"""
        pass
    
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
        
        # Attempt JSON serialization
        try:
            return json.dumps(matrix)
        except:
            # If not JSON serializable, use pickle as a fallback
            return pickle.dumps(matrix).hex()