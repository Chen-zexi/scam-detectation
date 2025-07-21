#!/usr/bin/env python3
"""
Database Module

This module provides MongoDB integration for the scam detection project.
"""

from .mongodb_config import (
    MongoDBConfig,
    MongoDBConnection,
    get_mongodb_connection,
    test_connection
)

from .scam_data_service import (
    ScamDataService,
    get_scam_data_service
)

__all__ = [
    'MongoDBConfig',
    'MongoDBConnection', 
    'get_mongodb_connection',
    'test_connection',
    'ScamDataService',
    'get_scam_data_service'
] 