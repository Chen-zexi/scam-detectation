#!/usr/bin/env python3
"""
Schema Builder for Dynamic Pydantic Model Generation

This module dynamically creates Pydantic models from JSON schema definitions,
enabling flexible synthesis of different scam types without hardcoding schemas.
"""

from typing import Dict, Any, Type, Optional, List, Union
from pydantic import BaseModel, Field, create_model
from enum import Enum
import json
from ..exceptions import SchemaValidationError


class SchemaBuilder:
    """
    Dynamically builds Pydantic models from JSON schema definitions.
    Supports various field types and validation rules.
    """
    
    def __init__(self):
        self.type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        self._enum_cache = {}
    
    def build_response_schema(self, schema_name: str, schema_def: Dict[str, Any]) -> Type[BaseModel]:
        """
        Build a Pydantic model from a JSON schema definition.
        
        Args:
            schema_name: Name for the generated model
            schema_def: Dictionary containing field definitions
            
        Returns:
            A dynamically created Pydantic model class
        """
        fields = {}
        
        for field_name, field_def in schema_def.items():
            field_type, field_attrs = self._parse_field_definition(field_name, field_def)
            fields[field_name] = (field_type, Field(**field_attrs))
        
        # Create and return the model
        return create_model(schema_name, **fields)
    
    def _parse_field_definition(self, field_name: str, field_def: Dict[str, Any]) -> tuple:
        """
        Parse a field definition and return the appropriate type and attributes.
        
        Args:
            field_name: Name of the field
            field_def: Field definition dictionary
            
        Returns:
            Tuple of (field_type, field_attributes)
        """
        # Get base type
        field_type_str = field_def.get("type", "string")
        field_type = self.type_mapping.get(field_type_str, str)
        
        # Handle enums
        if "enum" in field_def:
            enum_name = f"{field_name.capitalize()}Enum"
            if enum_name not in self._enum_cache:
                enum_values = {val.upper().replace(" ", "_"): val for val in field_def["enum"]}
                self._enum_cache[enum_name] = Enum(enum_name, enum_values)
            field_type = self._enum_cache[enum_name]
        
        # Handle nested objects
        elif field_type_str == "object" and "properties" in field_def:
            nested_model_name = f"{field_name.capitalize()}Model"
            field_type = self.build_response_schema(nested_model_name, field_def["properties"])
        
        # Handle arrays with specific item types
        elif field_type_str == "array" and "items" in field_def:
            item_type_str = field_def["items"].get("type", "string")
            item_type = self.type_mapping.get(item_type_str, str)
            field_type = List[item_type]
        
        # Build field attributes
        field_attrs = {
            "description": field_def.get("description", f"{field_name} field"),
            "default": field_def.get("default", ... if field_def.get("required", True) else None)
        }
        
        # Add validation attributes if present
        if "min_length" in field_def:
            field_attrs["min_length"] = field_def["min_length"]
        if "max_length" in field_def:
            field_attrs["max_length"] = field_def["max_length"]
        if "ge" in field_def:  # greater than or equal
            field_attrs["ge"] = field_def["ge"]
        if "le" in field_def:  # less than or equal
            field_attrs["le"] = field_def["le"]
        
        return field_type, field_attrs
    
    def validate_schema(self, schema_def: Dict[str, Any]) -> bool:
        """
        Validate that a schema definition is properly formatted.
        
        Args:
            schema_def: Schema definition to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not isinstance(schema_def, dict):
            raise SchemaValidationError("Schema definition must be a dictionary")
        
        for field_name, field_def in schema_def.items():
            if not isinstance(field_def, dict):
                raise SchemaValidationError(f"Field '{field_name}' definition must be a dictionary")
            
            if "type" not in field_def:
                raise SchemaValidationError(f"Field '{field_name}' must have a 'type' attribute")
            
            field_type = field_def["type"]
            if field_type not in self.type_mapping:
                raise SchemaValidationError(f"Unknown type '{field_type}' for field '{field_name}'. Supported types: {', '.join(self.type_map.keys())}")
            
            # Validate enum values
            if "enum" in field_def:
                if not isinstance(field_def["enum"], list):
                    raise SchemaValidationError(f"Enum values for field '{field_name}' must be a list")
                if len(field_def["enum"]) == 0:
                    raise SchemaValidationError(f"Enum values for field '{field_name}' cannot be empty")
        
        return True
    
    def get_schema_fields(self, model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Extract field information from a Pydantic model.
        
        Args:
            model: Pydantic model class
            
        Returns:
            Dictionary of field names to field info
        """
        fields = {}
        # Use model_fields for Pydantic v2 compatibility
        model_fields = getattr(model, 'model_fields', None) or getattr(model, '__fields__', {})
        
        for field_name, field_info in model_fields.items():
            fields[field_name] = {
                "type": str(field_info.annotation),
                "required": field_info.is_required() if hasattr(field_info, 'is_required') else True,
                "description": field_info.description if hasattr(field_info, 'description') else "",
                "default": field_info.default if hasattr(field_info, 'default') else None
            }
        return fields
    
    def create_example_instance(self, model: Type[BaseModel], category: str = "example") -> BaseModel:
        """
        Create an example instance of a model with default values.
        
        Args:
            model: Pydantic model class
            category: Category name for the example
            
        Returns:
            Instance of the model with example data
        """
        example_data = {}
        
        # Use model_fields for Pydantic v2 compatibility
        model_fields = getattr(model, 'model_fields', None) or getattr(model, '__fields__', {})
        
        for field_name, field_info in model_fields.items():
            field_type = field_info.annotation if hasattr(field_info, 'annotation') else field_info.type_
            
            # Generate example values based on field name and type
            if field_name == "timestamp":
                example_data[field_name] = "2024-01-01T00:00:00"
            elif field_name == "category":
                example_data[field_name] = category
            elif field_type == str:
                example_data[field_name] = f"Example {field_name}"
            elif field_type == int:
                example_data[field_name] = 100
            elif field_type == bool:
                example_data[field_name] = True
            elif hasattr(field_type, "__origin__") and field_type.__origin__ == list:
                example_data[field_name] = []
            elif hasattr(field_type, "_name") and "Enum" in str(field_type):
                # Get first enum value
                example_data[field_name] = list(field_type.__members__.values())[0].value
            else:
                example_data[field_name] = None
        
        return model(**example_data)