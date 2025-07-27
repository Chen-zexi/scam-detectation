#!/usr/bin/env python3
"""
Prompt Knowledge Base Service

This module provides service layer for managing scam knowledge
in MongoDB, including CRUD operations and useful query methods.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError
from pymongo import ASCENDING

from .mongodb_config import get_mongodb_connection, MongoDBConnection
from .knowledge_base_models import ScamKnowledge, KNOWLEDGE_BASE_COLLECTION, KNOWLEDGE_BASE_INDEXES

logger = logging.getLogger(__name__)


class PromptKnowledgeBaseService:
    """Simple service for managing scam knowledge base"""
    
    def __init__(self, connection: Optional[MongoDBConnection] = None):
        """
        Initialize the knowledge base service.
        
        Args:
            connection: MongoDB connection instance. If None, uses global connection.
        """
        self.connection = connection or get_mongodb_connection()
        self._collection_initialized = False
    
    def _ensure_collection_setup(self) -> bool:
        """
        Ensure collection is properly set up with indexes.
        
        Returns:
            True if setup successful, False otherwise
        """
        if self._collection_initialized:
            return True
        
        try:
            collection = self.connection.get_collection(KNOWLEDGE_BASE_COLLECTION)
            if collection is None:
                logger.error(f"Failed to get collection: {KNOWLEDGE_BASE_COLLECTION}")
                return False
            
            # Create indexes
            for index in KNOWLEDGE_BASE_INDEXES:
                if isinstance(index, tuple):
                    # Single field index
                    collection.create_index([(index[0], index[1])], unique=(index[0] == "id"))
                else:
                    # Compound index
                    collection.create_index(index)
            
            logger.info(f"Initialized knowledge base collection with indexes")
            self._collection_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error setting up knowledge base collection: {e}")
            return False
    
    def create_knowledge(self, knowledge: ScamKnowledge) -> Optional[str]:
        """
        Create a new scam knowledge entry.
        
        Args:
            knowledge: ScamKnowledge instance to create
            
        Returns:
            ID of created knowledge or None if failed
        """
        if not self._ensure_collection_setup():
            return None
        
        collection = self.connection.get_collection(KNOWLEDGE_BASE_COLLECTION)
        if collection is None:
            return None
        
        try:
            # Convert to dict and insert
            doc = knowledge.dict()
            result = collection.insert_one(doc)
            logger.info(f"Created knowledge entry: {knowledge.id}")
            return knowledge.id
            
        except DuplicateKeyError:
            logger.error(f"Knowledge with ID {knowledge.id} already exists")
            return None
        except Exception as e:
            logger.error(f"Error creating knowledge: {e}")
            return None
    
    def get_knowledge(self, knowledge_id: str) -> Optional[ScamKnowledge]:
        """
        Get a specific knowledge entry by ID.
        
        Args:
            knowledge_id: The ID to lookup
            
        Returns:
            ScamKnowledge instance or None if not found
        """
        if not self._ensure_collection_setup():
            return None
        
        collection = self.connection.get_collection(KNOWLEDGE_BASE_COLLECTION)
        if collection is None:
            return None
        
        try:
            doc = collection.find_one({"id": knowledge_id})
            if doc:
                # Remove MongoDB's _id field
                doc.pop('_id', None)
                return ScamKnowledge(**doc)
            return None
            
        except Exception as e:
            logger.error(f"Error getting knowledge {knowledge_id}: {e}")
            return None
    
    def get_knowledge_by_type(self, scam_type: str, 
                             classification: Optional[str] = None,
                             active_only: bool = True) -> List[ScamKnowledge]:
        """
        Get all knowledge entries for a specific scam type.
        
        Args:
            scam_type: The type to filter by (e.g., 'phone_transcript')
            classification: Optional classification filter
            active_only: Whether to return only active entries
            
        Returns:
            List of ScamKnowledge instances
        """
        if not self._ensure_collection_setup():
            return []
        
        collection = self.connection.get_collection(KNOWLEDGE_BASE_COLLECTION)
        if collection is None:
            return []
        
        try:
            # Build query
            query = {"type": scam_type}
            if classification:
                query["classification"] = classification
            if active_only:
                query["is_active"] = True
            
            # Execute query
            cursor = collection.find(query).sort("category", ASCENDING)
            results = []
            
            for doc in cursor:
                doc.pop('_id', None)
                results.append(ScamKnowledge(**doc))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting knowledge by type {scam_type}: {e}")
            return []
    
    def get_knowledge_by_category(self, category: str) -> Optional[ScamKnowledge]:
        """
        Get knowledge entry by category name.
        
        Args:
            category: The category to lookup
            
        Returns:
            ScamKnowledge instance or None if not found
        """
        if not self._ensure_collection_setup():
            return None
        
        collection = self.connection.get_collection(KNOWLEDGE_BASE_COLLECTION)
        if collection is None:
            return None
        
        try:
            doc = collection.find_one({"category": category, "is_active": True})
            if doc:
                doc.pop('_id', None)
                return ScamKnowledge(**doc)
            return None
            
        except Exception as e:
            logger.error(f"Error getting knowledge by category {category}: {e}")
            return None
    
    def search_by_tags(self, tags: List[str]) -> List[ScamKnowledge]:
        """
        Search knowledge entries by tags.
        
        Args:
            tags: List of tags to search for (OR operation)
            
        Returns:
            List of ScamKnowledge instances
        """
        if not self._ensure_collection_setup():
            return []
        
        collection = self.connection.get_collection(KNOWLEDGE_BASE_COLLECTION)
        if collection is None:
            return []
        
        try:
            # Find documents with any of the specified tags
            cursor = collection.find({
                "tags": {"$in": tags},
                "is_active": True
            })
            
            results = []
            for doc in cursor:
                doc.pop('_id', None)
                results.append(ScamKnowledge(**doc))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching by tags: {e}")
            return []
    
    def update_knowledge(self, knowledge_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a knowledge entry.
        
        Args:
            knowledge_id: ID of knowledge to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_collection_setup():
            return False
        
        collection = self.connection.get_collection(KNOWLEDGE_BASE_COLLECTION)
        if collection is None:
            return False
        
        try:
            # Add updated timestamp
            updates["updated_at"] = datetime.utcnow()
            
            result = collection.update_one(
                {"id": knowledge_id},
                {"$set": updates}
            )
            
            success = result.modified_count > 0
            if success:
                logger.info(f"Updated knowledge: {knowledge_id}")
            else:
                logger.warning(f"No knowledge found to update: {knowledge_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating knowledge {knowledge_id}: {e}")
            return False
    
    def delete_knowledge(self, knowledge_id: str, soft_delete: bool = True) -> bool:
        """
        Delete a knowledge entry.
        
        Args:
            knowledge_id: ID of knowledge to delete
            soft_delete: If True, only marks as inactive. If False, removes from DB
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_collection_setup():
            return False
        
        collection = self.connection.get_collection(KNOWLEDGE_BASE_COLLECTION)
        if collection is None:
            return False
        
        try:
            if soft_delete:
                # Soft delete - just mark as inactive
                return self.update_knowledge(knowledge_id, {"is_active": False})
            else:
                # Hard delete - remove from database
                result = collection.delete_one({"id": knowledge_id})
                success = result.deleted_count > 0
                
                if success:
                    logger.info(f"Deleted knowledge: {knowledge_id}")
                else:
                    logger.warning(f"No knowledge found to delete: {knowledge_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Error deleting knowledge {knowledge_id}: {e}")
            return False
    
    def get_all_types(self) -> List[str]:
        """
        Get all unique scam types in the knowledge base.
        
        Returns:
            List of unique scam types
        """
        if not self._ensure_collection_setup():
            return []
        
        collection = self.connection.get_collection(KNOWLEDGE_BASE_COLLECTION)
        if collection is None:
            return []
        
        try:
            types = collection.distinct("type", {"is_active": True})
            return sorted(types)
            
        except Exception as e:
            logger.error(f"Error getting all types: {e}")
            return []
    
    def get_categories_for_type(self, scam_type: str) -> List[Dict[str, str]]:
        """
        Get all categories for a specific scam type.
        
        Args:
            scam_type: The scam type to get categories for
            
        Returns:
            List of dictionaries with category info
        """
        if not self._ensure_collection_setup():
            return []
        
        collection = self.connection.get_collection(KNOWLEDGE_BASE_COLLECTION)
        if collection is None:
            return []
        
        try:
            pipeline = [
                {"$match": {"type": scam_type, "is_active": True}},
                {"$project": {
                    "category": 1,
                    "name": 1,
                    "classification": 1
                }},
                {"$sort": {"category": 1}}
            ]
            
            cursor = collection.aggregate(pipeline)
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Error getting categories for type {scam_type}: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        if not self._ensure_collection_setup():
            return {}
        
        collection = self.connection.get_collection(KNOWLEDGE_BASE_COLLECTION)
        if collection is None:
            return {}
        
        try:
            # Total count
            total = collection.count_documents({})
            active = collection.count_documents({"is_active": True})
            
            # Count by type
            type_pipeline = [
                {"$group": {
                    "_id": "$type",
                    "count": {"$sum": 1},
                    "active_count": {
                        "$sum": {"$cond": ["$is_active", 1, 0]}
                    }
                }}
            ]
            type_counts = list(collection.aggregate(type_pipeline))
            
            # Count by classification
            class_pipeline = [
                {"$group": {
                    "_id": "$classification",
                    "count": {"$sum": 1}
                }}
            ]
            class_counts = list(collection.aggregate(class_pipeline))
            
            return {
                "total_knowledge": total,
                "active_knowledge": active,
                "by_type": {item["_id"]: item for item in type_counts},
                "by_classification": {item["_id"]: item["count"] for item in class_counts}
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {}


# Global service instance
_global_kb_service: Optional[PromptKnowledgeBaseService] = None


def get_knowledge_base_service() -> PromptKnowledgeBaseService:
    """
    Get or create global knowledge base service instance.
    
    Returns:
        PromptKnowledgeBaseService instance
    """
    global _global_kb_service
    if _global_kb_service is None:
        _global_kb_service = PromptKnowledgeBaseService()
    return _global_kb_service