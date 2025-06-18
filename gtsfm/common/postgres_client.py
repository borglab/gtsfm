"""PostgreSQL database client module for GTSFM.

This module provides the PostgresClient class which handles database connections 
and operations for the GTSFM pipeline. It abstracts database interaction details
and provides a clean interface for executing queries and managing connections.

Authors: Zongyue Liu
"""
import logging
import psycopg2

from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class PostgresClient:
    """PostgreSQL database client for handling connections and operations"""
    
    def __init__(self, db_params: Dict[str, Any]) -> None:
        """
        Initialize the PostgreSQL client
        
        Args:
            db_params: Database connection parameters including host, port, database, user, and password
            
        Returns:
            None
        """
        self.db_params = db_params
        self.conn = None
        self.cursor = None
        self._schema_initialized = False
    
    def connect(self) -> bool:
        """
        Establish a database connection
        
        Args:
            None
        
        Returns:
            bool: True if the connection was successfully established, False otherwise
        """
        try:
            if self.conn is None or self.conn.closed:
                self.conn = psycopg2.connect(**self.db_params)
                self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def close(self) -> None:
        """
        Close the database connection
        
        Args:
            None
            
        Returns:
            None
        """
        try:
            if self.cursor:
                self.cursor.close()
                self.cursor = None
            if self.conn:
                self.conn.close()
                self.conn = None
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")
    
    def execute(self, query: str, params: Optional[Tuple] = None) -> bool:
        """
        Execute an SQL query with fresh connection (distributed-safe)
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            bool: Whether the execution was successful
        """
        try:
            # Use fresh connection like execute_with_connection
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return False
    
    def fetch_all(self, query: str, params: Optional[Tuple] = None) -> Optional[List[Tuple]]:
        """
        Execute a query and fetch all results
        
        Args:
            query (str): SQL query string
            params (tuple, optional): Query parameters
            
        Returns:
            list: List of query results, or None if failed
        """
        if not self.connect():
            return None
         
        try:
            self.cursor.execute(query, params)
            result = self.cursor.fetchall()
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None
        finally:
            self.close()
    
    def fetch_one(self, query: str, params: Optional[Tuple] = None) -> Optional[Tuple]:
        """
        Execute a query and fetch one result
        
        Args:
            query (str): SQL query string
            params (tuple, optional): Query parameters
            
        Returns:
            tuple: Single query result, or None if failed
        """
        if not self.connect():
            return None
        
        try:
            self.cursor.execute(query, params)
            result = self.cursor.fetchone()
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None
        finally:
            self.close()
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database
        
        Args:
            table_name (str): Name of the table to check
            
        Returns:
            bool: True if table exists, False otherwise
        """
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        );
        """
        result = self.fetch_one(query, (table_name,))
        return result[0] if result else False
    
    def create_table_if_not_exists(self, table_name: str, create_query: str) -> bool:
        """
        Create a table if it doesn't exist
        
        Args:
            table_name (str): Name of the table
            create_query (str): SQL CREATE TABLE statement
            
        Returns:
            bool: True if successful
        """
        if not self.table_exists(table_name):
            return self.execute(create_query)
        return True
    
    def __getstate__(self) -> Dict[str, Any]:
        """
        Custom serialization to avoid serializing connection objects
        
        Args:
            None
            
        Returns:
            Dict[str, Any]: Dictionary containing the object state with connection objects removed
        """
        state = self.__dict__.copy()
        # Do not serialize connection and cursor
        state['conn'] = None
        state['cursor'] = None
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Custom deserialization
        
        Args:
            state: Dictionary containing the object state to restore
            
        Returns:
            None
        """
        self.__dict__.update(state)
        # Connection will be re-established when needed
    
    
    def execute_with_schema_check(self, query: str, params: Optional[Tuple] = None) -> bool:
        """
        Execute query (schema checking handled by individual modules)
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            bool: True if query execution is successful, False otherwise
        """
        return self.execute(query, params)

    def execute_with_connection(self, query: str, params: Optional[Tuple] = None) -> bool:
        """
        Execute query with proper connection management
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            bool: True if query execution is successful, False otherwise
        """
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return False
