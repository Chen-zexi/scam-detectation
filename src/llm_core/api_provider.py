import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from ..exceptions import MissingAPIKeyError, InvalidProviderError

load_dotenv()


class ModelConfig:
    """Manages model configuration from JSON file."""
    
    def __init__(self):
        config_path = Path(__file__).parent / "model_config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def get_models(self, provider: str) -> List[Dict]:
        """Get list of models for a provider."""
        return self.config["models"].get(provider, [])
    
    def get_model_info(self, provider: str, model_id: str) -> Optional[Dict]:
        """Get information about a specific model."""
        models = self.get_models(provider)
        for model in models:
            if model["id"] == model_id:
                return model
        return None
    
    def get_model_parameters(self, provider: str, model_id: str) -> Dict[str, Any]:
        """Get default parameters for a specific model."""
        model_info = self.get_model_info(provider, model_id)
        if model_info:
            # Extract defaults from parameter definitions
            defaults = {}
            parameters = model_info.get("parameters", {})
            for param_name, param_def in parameters.items():
                if "default" in param_def:
                    defaults[param_name] = param_def["default"]
            return defaults
        
        # For dynamic models (lm-studio, vllm), use provider defaults
        provider_config = self.get_provider_config(provider)
        return provider_config.get("default_parameters", {"temperature": 0})
    
    def get_supported_parameters(self, provider: str, model_id: str) -> List[str]:
        """Get list of supported parameters for a model."""
        model_info = self.get_model_info(provider, model_id)
        if model_info and "parameters" in model_info:
            return list(model_info["parameters"].keys())
        return []
    
    def get_unsupported_parameters(self, provider: str, model_id: str) -> List[str]:
        """Get list of unsupported parameters for a model."""
        model_info = self.get_model_info(provider, model_id)
        if model_info:
            return model_info.get("unsupported_parameters", [])
        return []
    
    def is_reasoning_model(self, provider: str, model_id: str) -> bool:
        """Check if a model is a reasoning model."""
        model_info = self.get_model_info(provider, model_id)
        return model_info.get("is_reasoning", False) if model_info else False
    
    def get_provider_config(self, provider: str) -> Dict:
        """Get provider configuration."""
        return self.config["provider_config"].get(provider, {})
    
    def get_model_pricing(self, provider: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Get pricing information for a specific model.
        
        Returns:
            Dictionary with pricing info or None if not available
        """
        model_info = self.get_model_info(provider, model_id)
        if model_info and 'pricing' in model_info:
            return model_info['pricing']
        return None


class LLM:
    """A factory class for creating LangChain LLM clients for various providers."""
    
    # Class-level model config instance
    _model_config = None
    
    @classmethod
    def get_model_config(cls) -> ModelConfig:
        """Get or create the model configuration singleton."""
        if cls._model_config is None:
            cls._model_config = ModelConfig()
        return cls._model_config
    
    def __init__(self, provider: str, model: str, use_response_api: bool = False, **kwargs):
        """
        Initializes the LLM factory.

        Args:
            provider: The name of the LLM provider (e.g., 'openai', 'lm-studio').
            model: The specific model name to use.
            use_response_api: Whether to use OpenAI's Response API (output_version="responses/v1")
            **kwargs: Additional parameters to override defaults.
        """
        self.provider = provider
        self.model = model
        self.use_response_api = use_response_api
        self.model_config = self.get_model_config()
        
        # Get model-specific default parameters
        self.parameters = self.model_config.get_model_parameters(provider, model)
        
        # Override with any provided parameters
        self.parameters.update(kwargs)
        
        # Filter out unsupported parameters
        unsupported = self.model_config.get_unsupported_parameters(provider, model)
        for param in unsupported:
            self.parameters.pop(param, None)

    def _prepare_openai_params(self) -> Dict[str, Any]:
        """Prepare parameters for OpenAI models."""
        params = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": self.model,
            "stream_usage": True  # Enable token usage tracking by default
        }
        
        if not params["api_key"]:
            raise MissingAPIKeyError("OPENAI_API_KEY environment variable is not set")
        
        # Handle Response API if requested
        if self.use_response_api:
            # Add output_version parameter for Response API
            params["model_kwargs"] = {
                "output_version": "responses/v1"
            }
        
        # Handle model-specific parameters
        model_info = self.model_config.get_model_info(self.provider, self.model)
        
        if model_info and model_info.get("is_reasoning"):
            # For reasoning models (GPT-5, O3, O4-mini, etc.)
            
            # Build the reasoning parameter structure
            if "reasoning_effort" in self.parameters:
                # Pass reasoning as a direct parameter with nested structure
                params["reasoning"] = {
                    "effort": self.parameters["reasoning_effort"]
                }
            
            # Handle max_completion_tokens for O-series models
            if "max_completion_tokens" in self.parameters:
                params["max_tokens"] = self.parameters["max_completion_tokens"]
                
        else:
            # For standard models, apply normal parameters
            if "temperature" in self.parameters:
                params["temperature"] = self.parameters["temperature"]
            if "top_p" in self.parameters:
                params["top_p"] = self.parameters["top_p"]
            if "presence_penalty" in self.parameters:
                params["presence_penalty"] = self.parameters["presence_penalty"]
            if "frequency_penalty" in self.parameters:
                params["frequency_penalty"] = self.parameters["frequency_penalty"]
            if "max_tokens" in self.parameters:
                params["max_tokens"] = self.parameters["max_tokens"]
        
        return params

    def _prepare_anthropic_params(self) -> Dict[str, Any]:
        """Prepare parameters for Anthropic models."""
        params = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": self.model
        }
        
        if not params["api_key"]:
            raise MissingAPIKeyError("ANTHROPIC_API_KEY environment variable is not set")
        
        # Apply standard parameters
        if "temperature" in self.parameters:
            params["temperature"] = self.parameters["temperature"]
        if "max_tokens" in self.parameters:
            params["max_tokens"] = self.parameters["max_tokens"]
        if "top_p" in self.parameters:
            params["top_p"] = self.parameters["top_p"]
        
        # Handle thinking parameter for Anthropic models
        # Default is disabled unless explicitly enabled
        thinking_config = self.parameters.get("thinking", None)
        
        if thinking_config:
            # If thinking is a dict, use it as-is
            if isinstance(thinking_config, dict):
                params["thinking"] = thinking_config
            # If thinking is a boolean or string
            elif thinking_config in [True, "enabled", "true"]:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 10000  # Default budget
                }
            else:
                params["thinking"] = {
                    "type": "disabled"
                }
        else:
            # Explicitly disable thinking by default
            params["thinking"] = {
                "type": "disabled"
            }
        
        return params

    def _prepare_gemini_params(self) -> Dict[str, Any]:
        """Prepare parameters for Gemini models."""
        params = {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": self.model
        }
        
        if not params["api_key"]:
            raise MissingAPIKeyError("GEMINI_API_KEY environment variable is not set")
        
        # Apply standard parameters
        if "temperature" in self.parameters:
            params["temperature"] = self.parameters["temperature"]
        
        if "top_p" in self.parameters:
            params["top_p"] = self.parameters["top_p"]
        
        if "max_output_tokens" in self.parameters:
            params["max_tokens"] = self.parameters["max_output_tokens"]
        
        # Handle thinking_budget for Gemini 2.5 models
        model_info = self.model_config.get_model_info(self.provider, self.model)
        if model_info and model_info.get("is_reasoning"):
            # Default to 0 (disabled) unless explicitly set
            thinking_budget = self.parameters.get("thinking_budget", 0)
            
            # Pass thinking configuration through generation_config
            # Note: This might need adjustment based on actual Gemini API implementation
            params["generation_config"] = {
                "thinking_budget": thinking_budget
            }
            
            # Also include other generation config parameters if present
            if "temperature" in params:
                params["generation_config"]["temperature"] = params.pop("temperature")
            if "top_p" in params:
                params["generation_config"]["top_p"] = params.pop("top_p")
            if "max_tokens" in params:
                params["generation_config"]["max_output_tokens"] = params.pop("max_tokens")
        
        return params

    def _prepare_local_params(self, base_url: str, api_key: str) -> Dict[str, Any]:
        """Prepare parameters for local models (LM-Studio, vLLM)."""
        params = {
            "base_url": base_url,
            "api_key": api_key,
            "model": self.model
        }
        
        # Apply standard parameters (usually just temperature)
        if "temperature" in self.parameters:
            params["temperature"] = self.parameters["temperature"]
        
        return params

    def get_llm(self):
        """
        Initializes and returns a LangChain LLM client for the configured provider.

        Returns:
            A LangChain chat model instance.

        Raises:
            MissingAPIKeyError: If required API keys are not set.
            InvalidProviderError: If the provider is unsupported.
        """
        if self.provider == "openai":
            params = self._prepare_openai_params()
            
            # Extract special parameters that need to go in model_kwargs
            reasoning_param = params.pop("reasoning", None)
            existing_model_kwargs = params.pop("model_kwargs", {})
            
            # Combine model_kwargs if we have reasoning params or Response API
            if reasoning_param or existing_model_kwargs:
                combined_model_kwargs = existing_model_kwargs.copy()
                if reasoning_param:
                    combined_model_kwargs["reasoning"] = reasoning_param
                
                # Use model_kwargs for special parameters
                # Suppress the warning about parameters in model_kwargs
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Parameters .* should be specified explicitly")
                    llm = ChatOpenAI(**params, model_kwargs=combined_model_kwargs)
            else:
                # Standard case without special parameters
                llm = ChatOpenAI(**params)
            
            return llm
            
        elif self.provider == "anthropic":
            params = self._prepare_anthropic_params()
            return ChatAnthropic(**params)
            
        elif self.provider == "gemini":
            params = self._prepare_gemini_params()
            return ChatGoogleGenerativeAI(**params)
            
        elif self.provider == "lm-studio":
            host_ip = os.getenv("HOST_IP")
            if not host_ip:
                raise MissingAPIKeyError("HOST_IP environment variable is not set for LM-Studio")
            params = self._prepare_local_params(f"http://{host_ip}:1234/v1", "lm-studio")
            return ChatOpenAI(**params)
            
        elif self.provider == "vllm":
            host_ip = os.getenv("HOST_IP")
            if not host_ip:
                raise MissingAPIKeyError("HOST_IP environment variable is not set for vLLM")
            print(f"http://{host_ip}:8000/v1")
            params = self._prepare_local_params(f"http://{host_ip}:8000/v1", "EMPTY")
            return ChatOpenAI(**params)
            
        else:
            raise InvalidProviderError(f"Unsupported provider: {self.provider}. Supported: openai, anthropic, gemini, lm-studio, vllm")
        
    def get_structure_model(self, provider: str = None):
        """Get a model for structure parsing. Defaults to OpenAI gpt-4.1-nano with Response API."""
        # Always use OpenAI gpt-4.1-nano for structure parsing by default
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise MissingAPIKeyError("OPENAI_API_KEY is not set (required for structure model parsing)")
        # Use Response API for better performance
        return ChatOpenAI(
            api_key=api_key, 
            model="gpt-4.1-nano", 
            temperature=0, 
            stream_usage=True,
            model_kwargs={"output_version": "responses/v1"}
        )
        
    def get_structure_model_legacy(self, provider: str):
        """Legacy method for getting provider-specific structure models."""
        if provider == "lm-studio":
            host_ip = os.getenv("HOST_IP")
            if not host_ip:
                raise MissingAPIKeyError("HOST_IP is not set for legacy LM-Studio structure model")
            return ChatOpenAI(base_url=f"http://{host_ip}:1234/v1", api_key='lm-studio', model='osmosis-structure-0.6b@f16', temperature=0)
        elif provider == "vllm":
            # Assume using lm-studio
            host_ip = os.getenv("HOST_IP")
            if not host_ip:
                raise MissingAPIKeyError("HOST_IP is not set for legacy vLLM structure model")
            print(f"http://{host_ip}:8000/v1")
            return ChatOpenAI(base_url=f"http://{host_ip}:8000/v1", api_key='EMPTY', model='osmosis-structure-0.6b@f16', temperature=0)
        else:
            raise InvalidProviderError(f"Unsupported provider for structure model: {provider}")