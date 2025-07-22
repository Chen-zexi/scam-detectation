#!/usr/bin/env python3
"""
Synthesis Prompts Manager

This module loads and manages prompts from the JSON configuration file,
providing a centralized way to access synthesis prompts and configurations.
"""

import json
import os
from typing import Dict, Any, Optional, List, Callable, TypeVar
from pathlib import Path
import logging
from functools import wraps

from ..database.knowledge_base_service import get_knowledge_base_service
from ..database.knowledge_base_models import ScamKnowledge
from ..exceptions import ConfigurationError, UnknownSynthesisTypeError, InvalidCategoryError

logger = logging.getLogger(__name__)

T = TypeVar('T')


def with_database_fallback(json_method_name: str):
    """
    Decorator that provides automatic database fallback to JSON methods.
    
    Args:
        json_method_name: Name of the JSON-based fallback method
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            if self.kb_service and self.use_database:
                try:
                    result = func(self, *args, **kwargs)
                    if result:  # Only return if we got a valid result
                        return result
                except Exception as e:
                    logger.warning(f"Database error in {func.__name__}: {e}")
            
            # Fall back to JSON method
            json_method = getattr(self, json_method_name)
            return json_method(*args, **kwargs)
        return wrapper
    return decorator


class SynthesisPromptsManager:
    """
    Manages synthesis prompts and configurations with support for both JSON files and MongoDB.
    Falls back to JSON if database is unavailable.
    """
    
    def __init__(self, config_path: str = None, use_database: bool = True):
        """
        Initialize the prompts manager.
        
        Args:
            config_path: Path to the synthesis configuration JSON file.
                        Defaults to config/synthesis_config.json
            use_database: Whether to try using database first (default: True)
        """
        if config_path is None:
            # Get the project root directory (2 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "synthesis_config.json"
        
        self.config_path = Path(config_path)
        self.use_database = use_database
        self.kb_service = None
        
        # Try to initialize database service if enabled
        if self.use_database:
            try:
                self.kb_service = get_knowledge_base_service()
                # Test connection
                if self.kb_service.get_all_types():
                    logger.info("Using MongoDB knowledge base for prompts")
                else:
                    logger.warning("MongoDB knowledge base is empty, falling back to JSON")
                    self.kb_service = None
            except Exception as e:
                logger.warning(f"Could not connect to MongoDB knowledge base: {e}")
                logger.info("Falling back to JSON configuration")
                self.kb_service = None
        
        # Load JSON config as fallback or primary source
        self.config = self._load_config()
        self.synthesis_types = self.config.get("synthesis_types", {})
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load the synthesis configuration from JSON file.
        
        Returns:
            Dictionary containing the configuration
        """
        if not self.config_path.exists():
            raise ConfigurationError(f"Synthesis configuration not found at {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in synthesis configuration: {e}") from e
    
    def reload_config(self):
        """Reload the configuration from file."""
        self.config = self._load_config()
        self.synthesis_types = self.config.get("synthesis_types", {})
    
    @with_database_fallback('_get_synthesis_types_from_json')
    def get_synthesis_types(self) -> List[str]:
        """
        Get list of available synthesis types.
        
        Returns:
            List of synthesis type identifiers
        """
        return self.kb_service.get_all_types() if self.kb_service else []
    
    def _get_synthesis_types_from_json(self) -> List[str]:
        """Get synthesis types from JSON configuration."""
        return list(self.synthesis_types.keys())
    
    def get_synthesis_type_info(self, synthesis_type: str) -> Dict[str, Any]:
        """
        Get complete information for a synthesis type.
        
        Args:
            synthesis_type: Type identifier (e.g., 'phone_transcript')
            
        Returns:
            Dictionary containing all synthesis type information
        """
        if synthesis_type not in self.synthesis_types:
            raise UnknownSynthesisTypeError(f"Unknown synthesis type: {synthesis_type}. Available types: {', '.join(self.synthesis_types.keys())}")
        
        return self.synthesis_types[synthesis_type]
    
    def get_response_schema(self, synthesis_type: str) -> Dict[str, Any]:
        """
        Get response schema for a synthesis type (for backward compatibility).
        Returns the LLM response schema.
        
        Args:
            synthesis_type: Type identifier
            
        Returns:
            Dictionary containing schema definition
        """
        type_info = self.get_synthesis_type_info(synthesis_type)
        # For backward compatibility, return llm_response_schema if it exists
        return type_info.get("llm_response_schema", type_info.get("response_schema", {}))
    
    def get_llm_response_schema(self, synthesis_type: str) -> Dict[str, Any]:
        """
        Get LLM response schema for a synthesis type.
        
        Args:
            synthesis_type: Type identifier
            
        Returns:
            Dictionary containing LLM schema definition
        """
        type_info = self.get_synthesis_type_info(synthesis_type)
        return type_info.get("llm_response_schema", {})
    
    def get_metadata_schema(self, synthesis_type: str) -> Dict[str, Any]:
        """
        Get metadata schema for a synthesis type.
        
        Args:
            synthesis_type: Type identifier
            
        Returns:
            Dictionary containing metadata schema definition
        """
        type_info = self.get_synthesis_type_info(synthesis_type)
        return type_info.get("metadata_schema", {})
    
    def get_full_schema(self, synthesis_type: str) -> Dict[str, Any]:
        """
        Get combined LLM response and metadata schema.
        
        Args:
            synthesis_type: Type identifier
            
        Returns:
            Dictionary containing full schema definition
        """
        llm_schema = self.get_llm_response_schema(synthesis_type)
        metadata_schema = self.get_metadata_schema(synthesis_type)
        # Combine both schemas
        return {**llm_schema, **metadata_schema}
    
    @with_database_fallback('_get_system_prompt_from_json')
    def get_system_prompt(self, synthesis_type: str) -> str:
        """
        Get system prompt for a synthesis type.
        
        Args:
            synthesis_type: Type identifier
            
        Returns:
            System prompt string
        """
        if not self.kb_service:
            return ""
            
        knowledge_entries = self.kb_service.get_knowledge_by_type(synthesis_type)
        if knowledge_entries and knowledge_entries[0].system_prompt:
            return knowledge_entries[0].system_prompt
        return ""
    
    def _get_system_prompt_from_json(self, synthesis_type: str) -> str:
        """Get system prompt from JSON configuration."""
        type_info = self.get_synthesis_type_info(synthesis_type)
        return type_info.get("system_prompt", "")
    
    @with_database_fallback('_get_categories_from_json')
    def get_categories(self, synthesis_type: str) -> Dict[str, Any]:
        """
        Get all categories for a synthesis type.
        
        Args:
            synthesis_type: Type identifier
            
        Returns:
            Dictionary of category definitions
        """
        if not self.kb_service:
            return {}
            
        knowledge_entries = self.kb_service.get_knowledge_by_type(synthesis_type)
        if not knowledge_entries:
            return {}
            
        categories = {}
        for entry in knowledge_entries:
            categories[entry.category] = {
                "name": entry.name,
                "classification": entry.classification,
                "prompt_template": entry.prompt,
                "description": entry.description
            }
        return categories
    
    def _get_categories_from_json(self, synthesis_type: str) -> Dict[str, Any]:
        """Get categories from JSON configuration."""
        type_info = self.get_synthesis_type_info(synthesis_type)
        return type_info.get("categories", {})
    
    def get_category_names(self, synthesis_type: str) -> List[str]:
        """
        Get list of category identifiers for a synthesis type.
        
        Args:
            synthesis_type: Type identifier
            
        Returns:
            List of category identifiers
        """
        categories = self.get_categories(synthesis_type)
        return list(categories.keys())
    
    def get_category_info(self, synthesis_type: str, category: str) -> Dict[str, Any]:
        """
        Get information for a specific category.
        
        Args:
            synthesis_type: Type identifier
            category: Category identifier
            
        Returns:
            Dictionary containing category information
        """
        categories = self.get_categories(synthesis_type)
        if category not in categories:
            raise InvalidCategoryError(f"Unknown category '{category}' for synthesis type '{synthesis_type}'. Available categories: {', '.join(categories.keys())}")
        
        return categories[category]
    
    @with_database_fallback('_get_prompt_for_category_from_json')
    def get_prompt_for_category(self, synthesis_type: str, category: str) -> str:
        """
        Get the prompt template for a specific category.
        
        Args:
            synthesis_type: Type identifier
            category: Category identifier
            
        Returns:
            Prompt template string
        """
        if not self.kb_service:
            return ""
            
        knowledge_id = f"{synthesis_type}.{category}"
        knowledge = self.kb_service.get_knowledge(knowledge_id)
        return knowledge.prompt if knowledge else ""
    
    def _get_prompt_for_category_from_json(self, synthesis_type: str, category: str) -> str:
        """Get prompt from JSON configuration."""
        category_info = self.get_category_info(synthesis_type, category)
        return category_info.get("prompt_template", "")
    
    def get_classification_for_category(self, synthesis_type: str, category: str) -> str:
        """
        Get the expected classification for a category.
        
        Args:
            synthesis_type: Type identifier
            category: Category identifier
            
        Returns:
            Classification string
        """
        category_info = self.get_category_info(synthesis_type, category)
        return category_info.get("classification", "UNKNOWN")
    
    def create_generation_prompt(self, synthesis_type: str, category: str) -> str:
        """
        Create a complete generation prompt combining system and category prompts.
        
        Args:
            synthesis_type: Type identifier
            category: Category identifier
            
        Returns:
            Complete prompt string
        """
        # Get components
        system_prompt = self.get_system_prompt(synthesis_type)
        category_info = self.get_category_info(synthesis_type, category)
        category_prompt = self.get_prompt_for_category(synthesis_type, category)
        
        # Build complete prompt
        prompt_parts = [
            system_prompt,
            "\n\nCATEGORY SPECIFIC INSTRUCTIONS:",
            f"Category: {category_info.get('name', category)}",
            f"Expected Classification: {category_info.get('classification', 'UNKNOWN')}",
            "\n" + category_prompt,
            "\nGenerate a realistic example following the guidelines above."
        ]
        
        return "\n".join(prompt_parts)
    
    def get_synthesis_type_by_name(self, name: str) -> Optional[str]:
        """
        Find synthesis type identifier by display name.
        
        Args:
            name: Display name to search for
            
        Returns:
            Synthesis type identifier or None if not found
        """
        for type_id, type_info in self.synthesis_types.items():
            if type_info.get("name", "").lower() == name.lower():
                return type_id
        return None
    
    def validate_category(self, synthesis_type: str, category: str) -> bool:
        """
        Check if a category is valid for a synthesis type.
        
        Args:
            synthesis_type: Type identifier
            category: Category identifier
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self.get_category_info(synthesis_type, category)
            return True
        except ValueError:
            return False
    
    def get_all_classifications(self, synthesis_type: str) -> List[str]:
        """
        Get all unique classifications used in a synthesis type.
        
        Args:
            synthesis_type: Type identifier
            
        Returns:
            List of unique classification values
        """
        categories = self.get_categories(synthesis_type)
        classifications = set()
        
        for category_info in categories.values():
            classification = category_info.get("classification")
            if classification:
                classifications.add(classification)
        
        return sorted(list(classifications))


# Legacy functions for backward compatibility
def load_synthesis_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load synthesis configuration from JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    manager = SynthesisPromptsManager(config_path)
    return manager.config


def get_prompt_for_type(synthesis_type: str, category: str, config_path: str = None) -> str:
    """
    Get prompt for a specific synthesis type and category.
    
    Args:
        synthesis_type: Type identifier
        category: Category identifier
        config_path: Optional path to config file
        
    Returns:
        Complete generation prompt
    """
    manager = SynthesisPromptsManager(config_path)
    return manager.create_generation_prompt(synthesis_type, category)


def get_system_prompt(synthesis_type: str, config_path: str = None) -> str:
    """
    Get system prompt for a synthesis type.
    
    Args:
        synthesis_type: Type identifier
        config_path: Optional path to config file
        
    Returns:
        System prompt string
    """
    manager = SynthesisPromptsManager(config_path)
    return manager.get_system_prompt(synthesis_type)