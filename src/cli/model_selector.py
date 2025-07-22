"""
Model provider selection and management.
"""

import os
import logging
import requests
from typing import List, Optional

logger = logging.getLogger(__name__)


class ModelSelector:
    """Handles model provider and model selection."""
    
    def __init__(self):
        """Initialize model selector with environment configuration."""
        self.openai_models = [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini"
        ]
        self.host_ip = os.getenv("HOST_IP", "127.0.0.1")
        
    def choose_provider(self) -> str:
        """
        Interactive provider selection.
        
        Returns:
            Selected provider name
        """
        print("\nSTEP 3: Select Provider")
        print("-" * 40)
        print("1. OpenAI (GPT models)")
        print("2. LM-Studio (Local models)")
        print("3. vLLM (High-performance local)")
        
        while True:
            try:
                choice = int(input("\nSelect provider (1-3): ").strip())
                if choice == 1:
                    print("Selected: OpenAI")
                    return "openai"
                elif choice == 2:
                    print("Selected: LM-Studio") 
                    return "lm-studio"
                elif choice == 3:
                    print("Selected: vLLM")
                    return "vllm"
                else:
                    print("Please enter 1, 2, or 3")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                raise SystemExit(0)
    
    def get_openai_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        return self.openai_models.copy()
    
    def get_lm_studio_models(self) -> List[str]:
        """
        Fetch available models from LM-Studio API.
        
        Returns:
            List of model names
        """
        models = []
        try:
            url = f"http://{self.host_ip}:1234/v1/models"
            response = requests.get(url, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    if model_id:
                        models.append(model_id)
            else:
                logger.warning(f"LM-Studio returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to LM-Studio at {self.host_ip}:1234")
            print(f"\nError: LM-Studio server not found at http://{self.host_ip}:1234")
            print("Please ensure LM-Studio is running with a loaded model.")
        except Exception as e:
            logger.error(f"Error fetching LM-Studio models: {e}")
            
        return models
    
    def get_vllm_models(self) -> List[str]:
        """
        Fetch available models from vLLM API.
        
        Returns:
            List of model names
        """
        models = []
        try:
            url = f"http://{self.host_ip}:8000/v1/models"
            response = requests.get(url, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    if model_id:
                        models.append(model_id)
            else:
                logger.warning(f"vLLM returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to vLLM at {self.host_ip}:8000")
            print(f"\nError: vLLM server not found at http://{self.host_ip}:8000")
            print("Please ensure vLLM is running with a loaded model.")
        except Exception as e:
            logger.error(f"Error fetching vLLM models: {e}")
            
        return models
    
    def choose_model(self, provider: str) -> str:
        """
        Interactive model selection for a provider.
        
        Args:
            provider: Selected provider name
            
        Returns:
            Selected model name
        """
        print(f"\nSTEP 4: Select Model ({provider.upper()})")
        print("-" * 40)
        
        if provider == "openai":
            models = self.get_openai_models()
        elif provider == "lm-studio":
            models = self.get_lm_studio_models()
        elif provider == "vllm":
            models = self.get_vllm_models()
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        if not models:
            print("No models available. Please check your provider configuration.")
            raise SystemExit(1)
        
        print("Available models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                choice = int(input(f"\nSelect model (1-{len(models)}): ").strip())
                if 1 <= choice <= len(models):
                    selected = models[choice - 1]
                    print(f"Selected: {selected}")
                    return selected
                else:
                    print(f"Please enter a number between 1 and {len(models)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                raise SystemExit(0)