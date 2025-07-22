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

from .base_mongodb_service import BaseMongoDBService

from .scam_data_service import (
    ScamDataService,
    get_scam_data_service
)

from .knowledge_base_models import (
    ScamKnowledge,
    KNOWLEDGE_BASE_COLLECTION
)

from .knowledge_base_service import (
    PromptKnowledgeBaseService,
    get_knowledge_base_service
)

__all__ = [
    'MongoDBConfig',
    'MongoDBConnection', 
    'get_mongodb_connection',
    'test_connection',
    'BaseMongoDBService',
    'ScamDataService',
    'get_scam_data_service',
    'ScamKnowledge',
    'KNOWLEDGE_BASE_COLLECTION',
    'PromptKnowledgeBaseService',
    'get_knowledge_base_service'
] 