#!/usr/bin/env python3
"""
MongoDB Schema Models for Scam Knowledge Base

This module defines schema for storing scam knowledge
including types, categories, and their associated prompts.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ScamKnowledge(BaseModel):
    """Model for all scam knowledge"""
    # Unique identifier using type.category format
    id: str = Field(..., description="Unique ID (e.g., 'phone_transcript.tech_support_scam')")
    
    # Core classification fields
    type: str = Field(..., description="Type of scam (e.g., 'phone_transcript', 'phishing_email')")
    category: str = Field(..., description="Category within type (e.g., 'tech_support_scam')")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Description of this scam type")
    classification: str = Field(..., description="Classification level (LEGITIMATE, SCAM, etc.)")
    
    # Prompt information
    prompt: str = Field(..., description="The prompt template for generating this type")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt override")
    
    # Simple metadata
    tags: List[str] = Field(default_factory=list, description="Tags for easy searching")
    is_active: bool = Field(default=True, description="Whether this knowledge is actively used")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional fields for flexibility
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional flexible metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Collection configuration
KNOWLEDGE_BASE_COLLECTION = "scam_knowledge_base"

# Indexes for optimal query performance
KNOWLEDGE_BASE_INDEXES = [
    ("id", 1),  # Unique index on ID
    ("type", 1),  # For filtering by type
    ("category", 1),  # For filtering by category
    ("classification", 1),  # For filtering by classification
    ("tags", 1),  # For tag-based searches
    ("is_active", 1),  # For active/inactive filtering
    [("type", 1), ("category", 1)],  # Compound index for type+category queries
    [("type", 1), ("classification", 1)],  # Compound index for type+classification
]