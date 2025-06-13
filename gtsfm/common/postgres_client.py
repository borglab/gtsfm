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
            db_params (dict): Database connection parameters including host, port, database, user, and password
        """
        self.db_params = db_params
        self.conn = None
        self.cursor = None
        self._schema_initialized = False
    
    def connect(self) -> bool:
        """Establish a database connection
        
        Returns:
            bool: True if the connection was successfully established
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
        """Close the database connection"""
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
        Execute an SQL query
        
        Args:
            query (str): SQL query string
            params (tuple, optional): Query parameters
            
        Returns:
            bool: Whether the execution was successful
        """
        if not self.connect():
            return False
        
        try:
            self.cursor.execute(query, params)
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self.close()
    
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
        """Custom serialization to avoid serializing connection objects"""
        state = self.__dict__.copy()
        # Do not serialize connection and cursor
        state['conn'] = None
        state['cursor'] = None
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Custom deserialization"""
        self.__dict__.update(state)
        # Connection will be re-established when needed
    
    def ensure_schema(self) -> bool:
        """Ensure database schema is initialized"""
        # Remove domain-specific schema initialization
        # Each module should handle its own schema
        return True
    
    def execute_with_schema_check(self, query: str, params: Optional[Tuple] = None) -> bool:
        """Execute query (schema checking handled by individual modules)"""
        return self.execute(query, params)
