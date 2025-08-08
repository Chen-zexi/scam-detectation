"""
Model Configuration Manager

Centralized utility for managing and formatting model configurations
across all task classes (evaluation, annotation, synthesis).
"""

from typing import Dict, Any, Optional


class ModelConfigManager:
    """Manages model configuration extraction and formatting."""
    
    @staticmethod
    def get_model_config(llm_instance: Any, 
                         provider: str, 
                         model: str,
                         enable_thinking: bool = False,
                         use_structure_model: bool = False,
                         **additional_params) -> Dict[str, Any]:
        """
        Extract comprehensive model configuration from LLM instance.
        
        Args:
            llm_instance: The LLM instance object
            provider: Provider name (e.g., 'openai', 'anthropic')
            model: Model name (e.g., 'gpt-5', 'o3')
            enable_thinking: Whether thinking/reasoning is enabled
            use_structure_model: Whether structure model is used
            **additional_params: Any additional task-specific parameters
            
        Returns:
            Dictionary containing complete model configuration
        """
        model_config = {
            'provider': provider,
            'model': model,
            'enable_thinking': enable_thinking,
            'use_structure_model': use_structure_model
        }
        
        # Add any additional task-specific parameters
        model_config.update(additional_params)
        
        # Get model-specific parameters from llm_instance if available
        if llm_instance:
            # Store raw parameters
            if hasattr(llm_instance, 'parameters'):
                model_config['parameters'] = llm_instance.parameters
                
                # Extract specific parameter values for easy access
                for param in ['reasoning_effort', 'verbosity', 'temperature', 
                             'max_completion_tokens', 'top_p', 'presence_penalty',
                             'frequency_penalty', 'max_tokens']:
                    if param in llm_instance.parameters:
                        model_config[param] = llm_instance.parameters[param]
            
            # Get model info from config if available
            if hasattr(llm_instance, 'model_config'):
                model_info = llm_instance.model_config.get_model_info(provider, model)
                if model_info:
                    model_config['is_reasoning'] = model_info.get('is_reasoning', False)
                    model_config['model_description'] = model_info.get('description', '')
                    
                    # Add supported/unsupported parameters info
                    model_config['supported_parameters'] = list(model_info.get('parameters', {}).keys())
                    model_config['unsupported_parameters'] = model_info.get('unsupported_parameters', [])
        
        return model_config
    
    @staticmethod
    def format_for_display(model_config: Dict[str, Any]) -> str:
        """
        Format model configuration for human-readable display.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Formatted string for display
        """
        lines = []
        lines.append("Model Configuration:")
        lines.append("-" * 40)
        
        # Basic information
        lines.append(f"Provider: {model_config.get('provider', 'N/A')}")
        lines.append(f"Model: {model_config.get('model', 'N/A')}")
        
        if 'model_description' in model_config:
            lines.append(f"Description: {model_config['model_description']}")
        
        # Model type
        if model_config.get('is_reasoning'):
            lines.append("Type: Reasoning Model")
        else:
            lines.append("Type: Standard Model")
        
        # Active parameters
        lines.append("\nActive Parameters:")
        param_mapping = {
            'reasoning_effort': 'Reasoning Effort',
            'verbosity': 'Verbosity',
            'temperature': 'Temperature',
            'max_completion_tokens': 'Max Completion Tokens',
            'top_p': 'Top-p',
            'presence_penalty': 'Presence Penalty',
            'frequency_penalty': 'Frequency Penalty',
            'max_tokens': 'Max Tokens'
        }
        
        for param_key, param_name in param_mapping.items():
            if param_key in model_config and model_config[param_key] is not None:
                lines.append(f"  - {param_name}: {model_config[param_key]}")
        
        # Task settings
        lines.append("\nTask Settings:")
        lines.append(f"  - Enable Thinking: {model_config.get('enable_thinking', False)}")
        lines.append(f"  - Use Structure Model: {model_config.get('use_structure_model', False)}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_for_csv(model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten model configuration for CSV export.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Flattened dictionary suitable for CSV
        """
        flattened = {
            'model_provider': model_config.get('provider', 'N/A'),
            'model_name': model_config.get('model', 'N/A'),
            'is_reasoning_model': model_config.get('is_reasoning', False),
            'model_description': model_config.get('model_description', 'N/A')
        }
        
        # Add parameter values with safe defaults
        param_fields = {
            'reasoning_effort': 'N/A',
            'verbosity': 'N/A',
            'temperature': 'N/A',
            'max_completion_tokens': 'N/A',
            'top_p': 'N/A',
            'presence_penalty': 'N/A',
            'frequency_penalty': 'N/A',
            'max_tokens': 'N/A'
        }
        
        for param, default in param_fields.items():
            flattened[f'param_{param}'] = model_config.get(param, default)
        
        # Add task settings
        flattened['enable_thinking'] = model_config.get('enable_thinking', False)
        flattened['use_structure_model'] = model_config.get('use_structure_model', False)
        
        return flattened
    
    @staticmethod
    def format_for_json(model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format model configuration for JSON export.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Structured dictionary for JSON serialization
        """
        json_config = {
            'model': {
                'provider': model_config.get('provider'),
                'name': model_config.get('model'),
                'description': model_config.get('model_description', ''),
                'is_reasoning': model_config.get('is_reasoning', False)
            },
            'parameters': {},
            'task_settings': {
                'enable_thinking': model_config.get('enable_thinking', False),
                'use_structure_model': model_config.get('use_structure_model', False)
            }
        }
        
        # Add active parameters
        param_keys = ['reasoning_effort', 'verbosity', 'temperature', 
                     'max_completion_tokens', 'top_p', 'presence_penalty',
                     'frequency_penalty', 'max_tokens']
        
        for param in param_keys:
            if param in model_config and model_config[param] is not None:
                json_config['parameters'][param] = model_config[param]
        
        # Add parameter metadata if available
        if 'supported_parameters' in model_config:
            json_config['model']['supported_parameters'] = model_config['supported_parameters']
        if 'unsupported_parameters' in model_config:
            json_config['model']['unsupported_parameters'] = model_config['unsupported_parameters']
        
        return json_config
    
    @staticmethod
    def get_parameter_summary(model_config: Dict[str, Any]) -> str:
        """
        Get a brief summary of key parameters for logging.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Brief parameter summary string
        """
        parts = []
        
        # Add model identifier
        parts.append(f"{model_config.get('provider', 'unknown')}/{model_config.get('model', 'unknown')}")
        
        # Add key parameters based on model type
        if model_config.get('is_reasoning'):
            if 'reasoning_effort' in model_config:
                parts.append(f"effort={model_config['reasoning_effort']}")
            if 'verbosity' in model_config:
                parts.append(f"verbosity={model_config['verbosity']}")
        else:
            if 'temperature' in model_config:
                parts.append(f"temp={model_config['temperature']}")
        
        return " | ".join(parts)