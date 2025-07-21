#!/usr/bin/env python3
"""
MongoDB Configuration and Connection Manager

This module provides configuration and connection management for MongoDB
integration with the scam detection project.
"""

import os
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging

logger = logging.getLogger(__name__)


class MongoDBConfig:
    """Configuration class for MongoDB connection settings."""
    
    def __init__(self):
        # Load from environment variables with sensible defaults
        self.host = os.getenv('MONGODB_HOST', 'localhost')
        self.port = int(os.getenv('MONGODB_PORT', '27017'))
        self.database_name = os.getenv('MONGODB_DATABASE', 'scam_detection')
        self.username = os.getenv('MONGODB_USERNAME')
        self.password = os.getenv('MONGODB_PASSWORD')
        self.auth_source = os.getenv('MONGODB_AUTH_SOURCE', 'admin')
        
        # Connection options
        self.server_selection_timeout_ms = int(os.getenv('MONGODB_TIMEOUT', '5000'))
        
    def get_connection_string(self) -> str:
        """
        Generate MongoDB connection string based on configuration.
        
        Returns:
            MongoDB connection string
        """
        if self.username and self.password:
            # Authenticated connection
            return (f"mongodb://{self.username}:{self.password}@"
                   f"{self.host}:{self.port}/{self.database_name}?"
                   f"authSource={self.auth_source}")
        else:
            # Local connection without authentication
            return f"mongodb://{self.host}:{self.port}/"


class MongoDBConnection:
    """MongoDB connection manager with automatic connection handling."""
    
    def __init__(self, config: Optional[MongoDBConfig] = None):
        """
        Initialize MongoDB connection manager.
        
        Args:
            config: MongoDB configuration. If None, creates default config.
        """
        self.config = config or MongoDBConfig()
        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            connection_string = self.config.get_connection_string()
            
            self._client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=self.config.server_selection_timeout_ms
            )
            
            # Test the connection
            self._client.admin.command('ping')
            
            # Get database
            self._database = self._client[self.config.database_name]
            
            logger.info(f"Successfully connected to MongoDB: {self.config.host}:{self.config.port}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self._client = None
            self._database = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            self._client = None
            self._database = None
            return False
    
    def disconnect(self):
        """Close MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("Disconnected from MongoDB")
    
    def get_database(self) -> Optional[Database]:
        """
        Get the database instance.
        
        Returns:
            Database instance if connected, None otherwise
        """
        if self._database is None:
            if not self.connect():
                return None
        return self._database
    
    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """
        Get a collection from the database.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection instance if connected, None otherwise
        """
        database = self.get_database()
        if database is not None:
            return database[collection_name]
        return None
    
    def is_connected(self) -> bool:
        """
        Check if connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        if self._client is None or self._database is None:
            return False
        
        try:
            self._client.admin.command('ping')
            return True
        except Exception:
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Global connection instance
_global_connection: Optional[MongoDBConnection] = None


def get_mongodb_connection() -> MongoDBConnection:
    """
    Get or create global MongoDB connection instance.
    
    Returns:
        MongoDBConnection instance
    """
    global _global_connection
    if _global_connection is None:
        _global_connection = MongoDBConnection()
    return _global_connection


def test_connection() -> bool:
    """
    Test MongoDB connection with current configuration.
    
    Returns:
        True if connection successful, False otherwise
    """
    connection = MongoDBConnection()
    success = connection.connect()
    connection.disconnect()
    return success 