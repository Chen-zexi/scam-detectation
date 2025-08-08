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
    
    def __init__(self, provider: str, model: str, **kwargs):
        """
        Initializes the LLM factory.

        Args:
            provider: The name of the LLM provider (e.g., 'openai', 'lm-studio').
            model: The specific model name to use.
            **kwargs: Additional parameters to override defaults.
        """
        self.provider = provider
        self.model = model
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
            "model": self.model
        }
        
        if not params["api_key"]:
            raise MissingAPIKeyError("OPENAI_API_KEY environment variable is not set")
        
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
            
            # Build the text parameter structure for GPT-5 models
            if "verbosity" in self.parameters:
                # Pass text as a direct parameter with nested structure
                params["text"] = {
                    "verbosity": self.parameters["verbosity"]
                }
            
            # Handle max_completion_tokens for O-series models
            if "max_completion_tokens" in self.parameters:
                params["max_tokens"] = self.parameters["max_completion_tokens"]
            
            # For reasoning models, we might need to use the responses API
            # Check if this is an O-series model
            if self.model in ["o3", "o4-mini", "o1-preview", "o1-mini"]:
                # These models might benefit from the responses API
                params["use_responses_api"] = True
                
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
            
            # For reasoning models, we might need special handling
            # Extract reasoning and text params if they exist
            reasoning_param = params.pop("reasoning", None)
            text_param = params.pop("text", None)
            use_responses_api = params.pop("use_responses_api", False)
            
            # Create the ChatOpenAI instance
            llm = ChatOpenAI(**params)
            
            # If we have reasoning or text params, we need to bind them
            # This is a workaround since ChatOpenAI might not directly support these yet
            if reasoning_param or text_param:
                extra_kwargs = {}
                if reasoning_param:
                    extra_kwargs["reasoning"] = reasoning_param
                if text_param:
                    extra_kwargs["text"] = text_param
                # Use model_kwargs for now until LangChain updates support
                # Suppress the warning about parameters in model_kwargs
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Parameters .* should be specified explicitly")
                    llm = ChatOpenAI(**params, model_kwargs=extra_kwargs)
            
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
        """Get a model for structure parsing. Defaults to OpenAI gpt-4.1-nano."""
        # Always use OpenAI gpt-4.1-nano for structure parsing by default
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise MissingAPIKeyError("OPENAI_API_KEY is not set (required for structure model parsing)")
        return ChatOpenAI(api_key=api_key, model="gpt-4.1-nano", temperature=0)
        
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