import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from ..exceptions import MissingAPIKeyError, InvalidProviderError

load_dotenv()


class LLM:
    """A factory class for creating LangChain LLM clients for various providers."""
    def __init__(self, provider: str, model: str):
        """
        Initializes the LLM factory.

        Args:
            provider: The name of the LLM provider (e.g., 'openai', 'lm-studio').
            model: The specific model name to use.
        """
        self.provider = provider
        self.model = model

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
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise MissingAPIKeyError("OPENAI_API_KEY environment variable is not set")
            return ChatOpenAI(api_key=api_key, model=self.model, temperature=0)
        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise MissingAPIKeyError("ANTHROPIC_API_KEY environment variable is not set")
            return ChatAnthropic(api_key=api_key, model=self.model, temperature=0)
        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise MissingAPIKeyError("GEMINI_API_KEY environment variable is not set")
            if self.model.startswith("gemini-2.5-flash"):
                return ChatGoogleGenerativeAI(api_key=api_key, model=self.model, temperature=0)
            else:
                return ChatGoogleGenerativeAI(api_key=api_key, model=self.model, temperature=0)
        elif self.provider == "lm-studio":
            host_ip = os.getenv("HOST_IP")
            if not host_ip:
                raise MissingAPIKeyError("HOST_IP environment variable is not set for LM-Studio")
            return ChatOpenAI(base_url=f"http://{host_ip}:1234/v1", api_key='lm-studio', model=self.model, temperature=0)
        elif self.provider == "vllm":
            # Assume using lm-studio
            host_ip = os.getenv("HOST_IP")
            if not host_ip:
                raise MissingAPIKeyError("HOST_IP environment variable is not set for vLLM")
            print(f"http://{host_ip}:8000/v1")
            return ChatOpenAI(base_url=f"http://{host_ip}:8000/v1", api_key='EMPTY', model=self.model, temperature=0)
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