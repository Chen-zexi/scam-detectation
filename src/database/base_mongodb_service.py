"""
Base MongoDB Service

This module provides a base class with common patterns for MongoDB services,
reducing code duplication and providing consistent error handling.
"""

from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import logging
from enum import Enum
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, DuplicateKeyError
from pymongo import IndexModel

from .mongodb_config import MongoDBConnection, get_mongodb_connection
from ..exceptions import DatabaseError, DatabaseOperationError

logger = logging.getLogger(__name__)


class BaseMongoDBService:
    """Base class for MongoDB services with common patterns."""
    
    def __init__(self, connection: Optional[MongoDBConnection] = None):
        """
        Initialize the base service.
        
        Args:
            connection: MongoDB connection instance. If None, uses global connection.
        """
        self.connection = connection or get_mongodb_connection()
        self._initialized_collections: Set[str] = set()
    
    def _ensure_collection_setup(self, collection_name: str, indexes: List[IndexModel] = None) -> bool:
        """
        Ensure collection is properly set up with indexes.
        
        Args:
            collection_name: Name of the collection
            indexes: List of index models to create
            
        Returns:
            True if setup successful, False otherwise
        """
        if collection_name in self._initialized_collections:
            return True
        
        try:
            collection = self.connection.get_collection(collection_name)
            if collection is None:
                logger.error(f"Failed to get collection: {collection_name}")
                return False
            
            # Create indexes if provided
            if indexes:
                try:
                    collection.create_indexes(indexes)
                    logger.info(f"Created indexes for collection: {collection_name}")
                except Exception as e:
                    logger.warning(f"Some indexes might already exist for {collection_name}: {e}")
            
            self._initialized_collections.add(collection_name)
            return True
            
        except Exception as e:
            logger.error(f"Error setting up collection {collection_name}: {e}")
            return False
    
    def _get_collection(self, collection_name: str) -> Optional[Collection]:
        """
        Get a collection with error handling.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection instance or None if error
        """
        try:
            collection = self.connection.get_collection(collection_name)
            if collection is None:
                logger.error(f"Failed to get collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Error getting collection {collection_name}: {e}")
            return None
    
    def _convert_enums_to_strings(self, obj: Any) -> Any:
        """
        Recursively convert enum values to strings in a data structure.
        
        Args:
            obj: The object to process (dict, list, or primitive value)
            
        Returns:
            Object with enums converted to strings
        """
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {key: self._convert_enums_to_strings(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_enums_to_strings(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_enums_to_strings(item) for item in obj)
        else:
            return obj
    
    def _prepare_timestamps(self, doc: Dict[str, Any], timestamp_fields: List[str] = None) -> Dict[str, Any]:
        """
        Convert timestamp strings to datetime objects for better querying.
        
        Args:
            doc: Document to process
            timestamp_fields: List of timestamp field names to convert
            
        Returns:
            Document with converted timestamps
        """
        if timestamp_fields is None:
            timestamp_fields = ['generation_timestamp', 'timestamp', 'created_at', 'updated_at']
        
        for field in timestamp_fields:
            if field in doc and isinstance(doc[field], str):
                try:
                    doc[f"{field}_iso"] = doc[field]  # Keep original
                    doc[field] = datetime.fromisoformat(doc[field].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    # Keep as string if parsing fails
                    pass
        
        return doc
    
    def _batch_insert(self, collection: Collection, documents: List[Dict[str, Any]], 
                     batch_size: int = 100, ordered: bool = False) -> Dict[str, Any]:
        """
        Insert documents in batches with error handling.
        
        Args:
            collection: MongoDB collection
            documents: List of documents to insert
            batch_size: Number of documents per batch
            ordered: Whether to stop on first error
            
        Returns:
            Dictionary with insertion statistics
        """
        if not documents:
            return {'success': True, 'inserted_count': 0, 'errors': []}
        
        inserted_count = 0
        errors = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                result = collection.insert_many(batch, ordered=ordered)
                inserted_count += len(result.inserted_ids)
                logger.info(f"Inserted batch of {len(result.inserted_ids)} documents")
                
            except BulkWriteError as e:
                # Handle partial success in bulk operations
                inserted_count += e.details.get('nInserted', 0)
                for error in e.details.get('writeErrors', []):
                    if error.get('code') != 11000:  # Not a duplicate key error
                        errors.append({
                            'index': error.get('index', 'unknown'),
                            'error': error.get('errmsg', 'Unknown error'),
                            'code': error.get('code')
                        })
                logger.warning(f"Bulk write error: {len(e.details.get('writeErrors', []))} errors")
                
            except Exception as e:
                errors.append({'error': f"Batch insert error: {str(e)}"})
                logger.error(f"Error inserting batch: {e}")
        
        return {
            'success': len(errors) == 0 or inserted_count > 0,
            'inserted_count': inserted_count,
            'total_documents': len(documents),
            'errors': errors
        }
    
    def _build_query_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build MongoDB query filter from simplified parameters.
        
        Args:
            filters: Dictionary of filter parameters
            
        Returns:
            MongoDB query filter
        """
        query = {}
        
        for key, value in filters.items():
            if value is None:
                continue
            
            # Handle list values as $in queries
            if isinstance(value, list):
                query[key] = {'$in': value}
            # Handle date ranges
            elif key.endswith('_from') and isinstance(value, (str, datetime)):
                field = key[:-5]  # Remove '_from'
                if isinstance(value, str):
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                query.setdefault(field, {})['$gte'] = value
            elif key.endswith('_to') and isinstance(value, (str, datetime)):
                field = key[:-3]  # Remove '_to'
                if isinstance(value, str):
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                query.setdefault(field, {})['$lte'] = value
            # Default equality
            else:
                query[key] = value
        
        return query
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self._get_collection(collection_name)
            if not collection:
                return {'error': f'Collection not found: {collection_name}'}
            
            stats = {
                'collection': collection_name,
                'count': collection.count_documents({}),
                'indexes': [index['name'] for index in collection.list_indexes()]
            }
            
            # Try to get storage stats (may require permissions)
            try:
                coll_stats = collection.database.command('collStats', collection_name)
                stats.update({
                    'size_bytes': coll_stats.get('size', 0),
                    'storage_size_bytes': coll_stats.get('storageSize', 0),
                    'avg_doc_size': coll_stats.get('avgObjSize', 0)
                })
            except Exception as e:
                logger.debug(f"Could not get storage stats for {collection_name}: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats for {collection_name}: {e}")
            return {'error': str(e)}