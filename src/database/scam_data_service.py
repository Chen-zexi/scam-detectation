#!/usr/bin/env python3
"""
Scam Data Service

This module provides a service layer for storing and retrieving scam detection
synthetic data in MongoDB. It handles different scam types and provides
optimized batch operations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from enum import Enum
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, DuplicateKeyError
from pymongo import IndexModel, ASCENDING, TEXT

from .mongodb_config import get_mongodb_connection, MongoDBConnection

logger = logging.getLogger(__name__)


class ScamDataService:
    """Service for storing and retrieving scam detection data in MongoDB."""
    
    # Collection names for different scam types
    COLLECTIONS = {
        'phone_transcript': 'phone_scams',
        'phishing_email': 'email_scams', 
        'sms_scam': 'sms_scams'
    }
    
    def __init__(self, connection: Optional[MongoDBConnection] = None):
        """
        Initialize the scam data service.
        
        Args:
            connection: MongoDB connection instance. If None, uses global connection.
        """
        self.connection = connection or get_mongodb_connection()
        self._initialized_collections = set()
    
    def _get_collection_name(self, synthesis_type: str) -> str:
        """
        Get collection name for a synthesis type.
        
        Args:
            synthesis_type: Type of synthesis (e.g., 'phone_transcript', 'phishing_email')
            
        Returns:
            Collection name
        """
        return self.COLLECTIONS.get(synthesis_type, f"{synthesis_type}_scams")
    
    def _ensure_collection_setup(self, collection_name: str) -> bool:
        """
        Ensure collection is properly set up with indexes.
        
        Args:
            collection_name: Name of the collection
            
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
            
            # Create indexes for better performance
            indexes = [
                IndexModel([("synthesis_type", ASCENDING)]),
                IndexModel([("classification", ASCENDING)]),
                IndexModel([("category", ASCENDING)]),
                IndexModel([("generation_timestamp", ASCENDING)]),
                IndexModel([("knowledge_id", ASCENDING)]),  # Link to knowledge base
                # Text index for content search (adjust field names as needed)
                IndexModel([("transcript", TEXT), ("content", TEXT), ("message", TEXT)])
            ]
            
            # Create indexes (ignore errors for existing indexes)
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
    
    def store_synthesis_results(self, 
                               synthesis_type: str, 
                               results: List[Dict[str, Any]], 
                               batch_size: int = 100,
                               knowledge_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Store synthesis results in MongoDB.
        
        Args:
            synthesis_type: Type of synthesis (e.g., 'phone_transcript', 'phishing_email')
            results: List of synthesis result dictionaries
            batch_size: Number of documents to insert per batch
            
        Returns:
            Dictionary with storage statistics
        """
        if not results:
            return {'success': True, 'inserted_count': 0, 'errors': []}
        
        collection_name = self._get_collection_name(synthesis_type)
        
        # Ensure collection is set up
        if not self._ensure_collection_setup(collection_name):
            return {'success': False, 'error': f'Failed to setup collection: {collection_name}'}
        
        collection = self.connection.get_collection(collection_name)
        if collection is None:
            return {'success': False, 'error': f'Failed to get collection: {collection_name}'}
        
        # Prepare documents for insertion
        documents = []
        for result in results:
            # Extract knowledge_id from result if not provided globally
            result_knowledge_id = knowledge_id
            if not result_knowledge_id and 'category' in result:
                result_knowledge_id = f"{synthesis_type}.{result['category']}"
            
            doc = self._prepare_document(result, synthesis_type, result_knowledge_id)
            if doc:
                documents.append(doc)
        
        if not documents:
            return {'success': True, 'inserted_count': 0, 'errors': ['No valid documents to insert']}
        
        # Insert in batches
        inserted_count = 0
        errors = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                result = collection.insert_many(batch, ordered=False)
                inserted_count += len(result.inserted_ids)
                logger.info(f"Inserted batch of {len(result.inserted_ids)} documents into {collection_name}")
                
            except BulkWriteError as e:
                # Handle partial success in bulk operations
                inserted_count += e.details.get('nInserted', 0)
                for error in e.details.get('writeErrors', []):
                    if error.get('code') != 11000:  # Not a duplicate key error
                        errors.append(f"Document {error.get('index', 'unknown')}: {error.get('errmsg', 'Unknown error')}")
                logger.warning(f"Bulk write error in {collection_name}: {len(e.details.get('writeErrors', []))} errors")
                
            except Exception as e:
                errors.append(f"Batch insert error: {str(e)}")
                logger.error(f"Error inserting batch into {collection_name}: {e}")
        
        logger.info(f"Stored {inserted_count} documents in {collection_name}")
        
        return {
            'success': len(errors) == 0 or inserted_count > 0,
            'inserted_count': inserted_count,
            'total_documents': len(documents),
            'collection': collection_name,
            'errors': errors
        }
    
    def _prepare_document(self, result: Dict[str, Any], synthesis_type: str, knowledge_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Prepare a result dictionary for MongoDB insertion.
        
        Args:
            result: Result dictionary from synthesis
            synthesis_type: Type of synthesis
            
        Returns:
            Document ready for insertion, or None if invalid
        """
        try:
            # Create a copy to avoid modifying the original
            doc = result.copy()
            
            # Convert enum values to strings recursively
            doc = self._convert_enums_to_strings(doc)
            
            # Ensure required fields
            doc['synthesis_type'] = synthesis_type
            doc['storage_timestamp'] = datetime.utcnow()
            
            # Add knowledge base link if provided
            if knowledge_id:
                doc['knowledge_id'] = knowledge_id
            elif 'category' in doc:
                # Auto-generate knowledge_id from type and category
                doc['knowledge_id'] = f"{synthesis_type}.{doc['category']}"
            
            # Ensure generation_timestamp exists
            if 'generation_timestamp' not in doc:
                doc['generation_timestamp'] = doc.get('timestamp', datetime.utcnow().isoformat())
            
            # Convert timestamp strings to datetime objects for better querying
            for timestamp_field in ['generation_timestamp', 'timestamp']:
                if timestamp_field in doc and isinstance(doc[timestamp_field], str):
                    try:
                        doc[f"{timestamp_field}_iso"] = doc[timestamp_field]  # Keep original
                        doc[timestamp_field] = datetime.fromisoformat(doc[timestamp_field].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        # Keep as string if parsing fails
                        pass
            
            # Remove 'id' field to avoid conflicts - MongoDB will use _id
            if 'id' in doc:
                del doc['id']
            
            return doc
            
        except Exception as e:
            logger.error(f"Error preparing document for insertion: {e}")
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
    
    def get_scam_data(self, 
                      synthesis_type: str = None, 
                      classification: str = None,
                      category: str = None,
                      knowledge_id: str = None,
                      limit: int = 100,
                      skip: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve scam data from MongoDB.
        
        Args:
            synthesis_type: Filter by synthesis type
            classification: Filter by classification
            category: Filter by category
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            
        Returns:
            List of documents
        """
        try:
            # If synthesis_type is specified, use that collection
            if synthesis_type:
                collection_name = self._get_collection_name(synthesis_type)
                collection = self.connection.get_collection(collection_name)
                if collection is None:
                    return []
                
                query = {'synthesis_type': synthesis_type}
                if classification:
                    query['classification'] = classification
                if category:
                    query['category'] = category
                if knowledge_id:
                    query['knowledge_id'] = knowledge_id
                
                cursor = collection.find(query).skip(skip).limit(limit).sort('generation_timestamp', -1)
                return list(cursor)
            
            else:
                # Search across all collections
                all_results = []
                for syn_type in self.COLLECTIONS.keys():
                    results = self.get_scam_data(syn_type, classification, category, knowledge_id, limit, skip)
                    all_results.extend(results)
                
                # Sort by generation timestamp and apply limit
                all_results.sort(key=lambda x: x.get('generation_timestamp', datetime.min), reverse=True)
                return all_results[:limit]
                
        except Exception as e:
            logger.error(f"Error retrieving scam data: {e}")
            return []
    
    def get_collection_stats(self, synthesis_type: str = None) -> Dict[str, Any]:
        """
        Get statistics about stored scam data.
        
        Args:
            synthesis_type: Specific synthesis type, or None for all
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        try:
            if synthesis_type:
                collection_name = self._get_collection_name(synthesis_type)
                collection = self.connection.get_collection(collection_name)
                if collection is not None:
                    total_count = collection.count_documents({})
                    stats[synthesis_type] = {
                        'total_documents': total_count,
                        'collection': collection_name
                    }
                    
                    # Get classification breakdown
                    pipeline = [
                        {'$group': {'_id': '$classification', 'count': {'$sum': 1}}}
                    ]
                    classification_stats = list(collection.aggregate(pipeline))
                    stats[synthesis_type]['classifications'] = {
                        item['_id']: item['count'] for item in classification_stats
                    }
            else:
                # Get stats for all synthesis types
                for syn_type in self.COLLECTIONS.keys():
                    type_stats = self.get_collection_stats(syn_type)
                    stats.update(type_stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        return self.connection.is_connected()


# Global service instance
_global_service: Optional[ScamDataService] = None


def get_scam_data_service() -> ScamDataService:
    """
    Get or create global scam data service instance.
    
    Returns:
        ScamDataService instance
    """
    global _global_service
    if _global_service is None:
        _global_service = ScamDataService()
    return _global_service 