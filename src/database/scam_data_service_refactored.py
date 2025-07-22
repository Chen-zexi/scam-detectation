#!/usr/bin/env python3
"""
Scam Data Service (Refactored)

This module provides a service layer for storing and retrieving scam detection
synthetic data in MongoDB. It extends the BaseMongoDBService for common patterns.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pymongo import IndexModel, ASCENDING, TEXT

from .base_mongodb_service import BaseMongoDBService
from .mongodb_config import MongoDBConnection

logger = logging.getLogger(__name__)


class ScamDataService(BaseMongoDBService):
    """Service for storing and retrieving scam detection data in MongoDB."""
    
    # Collection names for different scam types
    COLLECTIONS = {
        'phone_transcript': 'phone_scams',
        'phishing_email': 'email_scams', 
        'sms_scam': 'sms_scams'
    }
    
    # Default indexes for scam collections
    DEFAULT_INDEXES = [
        IndexModel([("id", ASCENDING)], unique=True),
        IndexModel([("synthesis_type", ASCENDING)]),
        IndexModel([("classification", ASCENDING)]),
        IndexModel([("category", ASCENDING)]),
        IndexModel([("generation_timestamp", ASCENDING)]),
        IndexModel([("knowledge_id", ASCENDING)]),
        # Text index for content search
        IndexModel([("transcript", TEXT), ("content", TEXT), ("message", TEXT)])
    ]
    
    def __init__(self, connection: Optional[MongoDBConnection] = None):
        """
        Initialize the scam data service.
        
        Args:
            connection: MongoDB connection instance. If None, uses global connection.
        """
        super().__init__(connection)
    
    def _get_collection_name(self, synthesis_type: str) -> str:
        """
        Get collection name for a synthesis type.
        
        Args:
            synthesis_type: Type of synthesis (e.g., 'phone_transcript', 'phishing_email')
            
        Returns:
            Collection name
        """
        return self.COLLECTIONS.get(synthesis_type, f"{synthesis_type}_scams")
    
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
            knowledge_id: Optional knowledge base ID to link
            
        Returns:
            Dictionary with storage statistics
        """
        if not results:
            return {'success': True, 'inserted_count': 0, 'errors': []}
        
        collection_name = self._get_collection_name(synthesis_type)
        
        # Ensure collection is set up with indexes
        if not self._ensure_collection_setup(collection_name, self.DEFAULT_INDEXES):
            return {'success': False, 'error': f'Failed to setup collection: {collection_name}'}
        
        collection = self._get_collection(collection_name)
        if not collection:
            return {'success': False, 'error': f'Failed to get collection: {collection_name}'}
        
        # Prepare documents for insertion
        documents = []
        for result in results:
            doc = self._prepare_document(result, synthesis_type, knowledge_id)
            if doc:
                documents.append(doc)
        
        if not documents:
            return {'success': True, 'inserted_count': 0, 'errors': ['No valid documents to insert']}
        
        # Use base class batch insert
        insert_result = self._batch_insert(collection, documents, batch_size)
        insert_result['collection'] = collection_name
        
        logger.info(f"Stored {insert_result['inserted_count']} documents in {collection_name}")
        
        return insert_result
    
    def _prepare_document(self, result: Dict[str, Any], synthesis_type: str, 
                         knowledge_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Prepare a result dictionary for MongoDB insertion.
        
        Args:
            result: Result dictionary from synthesis
            synthesis_type: Type of synthesis
            knowledge_id: Optional knowledge base ID
            
        Returns:
            Document ready for insertion, or None if invalid
        """
        try:
            # Create a copy to avoid modifying the original
            doc = result.copy()
            
            # Convert enum values to strings
            doc = self._convert_enums_to_strings(doc)
            
            # Ensure required fields
            doc['synthesis_type'] = synthesis_type
            doc['storage_timestamp'] = datetime.utcnow()
            
            # Add knowledge base link
            if knowledge_id:
                doc['knowledge_id'] = knowledge_id
            elif 'category' in doc:
                # Auto-generate knowledge_id from type and category
                doc['knowledge_id'] = f"{synthesis_type}.{doc['category']}"
            
            # Ensure generation_timestamp exists
            if 'generation_timestamp' not in doc:
                doc['generation_timestamp'] = doc.get('timestamp', datetime.utcnow().isoformat())
            
            # Convert timestamps
            doc = self._prepare_timestamps(doc)
            
            # Ensure we have an ID field
            if 'id' not in doc and '_id' not in doc:
                doc['id'] = hash(str(doc)) % (10**9)  # Generate a simple numeric ID
            
            return doc
            
        except Exception as e:
            logger.error(f"Error preparing document for insertion: {e}")
            return None
    
    def get_scam_data(self, 
                     synthesis_type: str = None,
                     classification: str = None,
                     category: str = None,
                     limit: int = 100,
                     skip: int = 0,
                     sort_by: str = "generation_timestamp",
                     sort_order: int = -1,
                     date_from: str = None,
                     date_to: str = None) -> Dict[str, Any]:
        """
        Retrieve scam data from MongoDB with filtering and pagination.
        
        Args:
            synthesis_type: Filter by synthesis type
            classification: Filter by classification
            category: Filter by category
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            sort_by: Field to sort by
            sort_order: Sort order (-1 for descending, 1 for ascending)
            date_from: Filter by generation timestamp (ISO format)
            date_to: Filter by generation timestamp (ISO format)
            
        Returns:
            Dictionary with results and metadata
        """
        try:
            # Determine collection(s) to query
            if synthesis_type:
                collection_names = [self._get_collection_name(synthesis_type)]
            else:
                collection_names = list(self.COLLECTIONS.values())
            
            all_results = []
            total_count = 0
            
            # Build query filter
            filters = {
                'synthesis_type': synthesis_type,
                'classification': classification,
                'category': category,
                'generation_timestamp_from': date_from,
                'generation_timestamp_to': date_to
            }
            query = self._build_query_filter(filters)
            
            # Query each collection
            for collection_name in collection_names:
                collection = self._get_collection(collection_name)
                if not collection:
                    continue
                
                # Get total count for this collection
                collection_count = collection.count_documents(query)
                total_count += collection_count
                
                # Skip if we've already collected enough from previous collections
                if skip >= len(all_results) + collection_count:
                    skip -= collection_count
                    continue
                
                # Calculate collection-specific skip and limit
                coll_skip = max(0, skip - len(all_results))
                coll_limit = min(limit - len(all_results), collection_count - coll_skip)
                
                if coll_limit <= 0:
                    continue
                
                # Query with pagination
                cursor = collection.find(query).sort(sort_by, sort_order).skip(coll_skip).limit(coll_limit)
                
                for doc in cursor:
                    # Convert ObjectId to string
                    if '_id' in doc:
                        doc['_id'] = str(doc['_id'])
                    all_results.append(doc)
                
                if len(all_results) >= limit:
                    break
            
            return {
                'success': True,
                'results': all_results,
                'total_count': total_count,
                'page_size': limit,
                'page_offset': skip,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Error retrieving scam data: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def get_stats_by_type(self, synthesis_type: str = None) -> Dict[str, Any]:
        """
        Get statistics for all collections or a specific synthesis type.
        
        Args:
            synthesis_type: Optional synthesis type to filter
            
        Returns:
            Dictionary with statistics
        """
        try:
            if synthesis_type:
                collection_names = [self._get_collection_name(synthesis_type)]
            else:
                collection_names = list(self.COLLECTIONS.values())
            
            stats = {
                'total_documents': 0,
                'by_collection': {},
                'by_classification': {},
                'by_category': {}
            }
            
            for collection_name in collection_names:
                # Get basic collection stats
                coll_stats = self.get_collection_stats(collection_name)
                if 'error' not in coll_stats:
                    stats['by_collection'][collection_name] = coll_stats
                    stats['total_documents'] += coll_stats.get('count', 0)
                
                # Get aggregated stats
                collection = self._get_collection(collection_name)
                if collection:
                    # Classification distribution
                    pipeline = [
                        {'$group': {'_id': '$classification', 'count': {'$sum': 1}}},
                        {'$sort': {'count': -1}}
                    ]
                    for item in collection.aggregate(pipeline):
                        classification = item['_id'] or 'Unknown'
                        stats['by_classification'][classification] = stats['by_classification'].get(classification, 0) + item['count']
                    
                    # Category distribution
                    pipeline = [
                        {'$group': {'_id': '$category', 'count': {'$sum': 1}}},
                        {'$sort': {'count': -1}}
                    ]
                    for item in collection.aggregate(pipeline):
                        category = item['_id'] or 'Unknown'
                        stats['by_category'][category] = stats['by_category'].get(category, 0) + item['count']
            
            return {
                'success': True,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Singleton instance
_scam_data_service = None


def get_scam_data_service(connection: Optional[MongoDBConnection] = None) -> ScamDataService:
    """
    Get the singleton ScamDataService instance.
    
    Args:
        connection: Optional MongoDB connection. Uses global connection if not provided.
        
    Returns:
        ScamDataService instance
    """
    global _scam_data_service
    if _scam_data_service is None or connection is not None:
        _scam_data_service = ScamDataService(connection)
    return _scam_data_service