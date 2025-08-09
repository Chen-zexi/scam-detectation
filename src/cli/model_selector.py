"""
Model provider selection and management with parameter configuration.
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class ModelSelector:
    """Handles model provider, model selection, and parameter configuration."""
    
    def __init__(self):
        """Initialize model selector with environment configuration."""
        # Load model configuration from JSON
        config_path = Path(__file__).parent.parent / "llm_core" / "model_config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
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
    
    def get_models_from_config(self, provider: str) -> List[Dict]:
        """
        Get models from configuration file.
        
        Args:
            provider: Provider name
            
        Returns:
            List of model dictionaries
        """
        return self.config["models"].get(provider, [])
    
    def get_openai_models(self) -> List[Dict]:
        """Get list of available OpenAI models from config."""
        return self.get_models_from_config("openai")
    
    def get_lm_studio_models(self) -> List[Dict]:
        """
        Fetch available models from LM-Studio API.
        
        Returns:
            List of model dictionaries
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
                        # Create a model dict compatible with our config format
                        models.append({
                            "id": model_id,
                            "name": model_id,
                            "is_reasoning": False,
                            "reasoning_efforts": [],
                            "description": "LM-Studio loaded model"
                        })
            else:
                logger.warning(f"LM-Studio returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to LM-Studio at {self.host_ip}:1234")
            print(f"\nError: LM-Studio server not found at http://{self.host_ip}:1234")
            print("Please ensure LM-Studio is running with a loaded model.")
        except Exception as e:
            logger.error(f"Error fetching LM-Studio models: {e}")
            
        return models
    
    def get_vllm_models(self) -> List[Dict]:
        """
        Fetch available models from vLLM API.
        
        Returns:
            List of model dictionaries
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
                        # Create a model dict compatible with our config format
                        models.append({
                            "id": model_id,
                            "name": model_id,
                            "is_reasoning": False,
                            "reasoning_efforts": [],
                            "description": "vLLM loaded model"
                        })
            else:
                logger.warning(f"vLLM returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to vLLM at {self.host_ip}:8000")
            print(f"\nError: vLLM server not found at http://{self.host_ip}:8000")
            print("Please ensure vLLM is running with a loaded model.")
        except Exception as e:
            logger.error(f"Error fetching vLLM models: {e}")
            
        return models
    
    def configure_model_parameters(self, provider: str, model_info: Dict) -> Dict[str, any]:
        """
        Configure parameters for the selected model.
        
        Args:
            provider: Provider name
            model_info: Model information dictionary
            
        Returns:
            Dictionary of configured parameters
        """
        parameters = {}
        
        # Check if this is a model with configurable parameters
        if model_info.get("parameters"):
            print("\n" + "="*40)
            print("MODEL PARAMETER CONFIGURATION")
            print("="*40)
            
            # Configure reasoning effort if available
            if "reasoning_effort" in model_info.get("parameters", {}):
                param_info = model_info["parameters"]["reasoning_effort"]
                options = param_info.get("options", [])
                default = param_info.get("default", options[0] if options else "medium")
                
                print(f"\nReasoning Effort:")
                print(f"Controls reasoning depth")
                for i, option in enumerate(options, 1):
                    default_marker = " (default)" if option == default else ""
                    print(f"  {i}. {option.capitalize()}{default_marker}")
                
                while True:
                    choice = input(f"\nSelect (1-{len(options)}) or press Enter for default [{default}]: ").strip()
                    if not choice:  # User pressed Enter
                        parameters['reasoning_effort'] = default
                        print(f"Using default: {default}")
                        break
                    try:
                        choice_num = int(choice)
                        if 1 <= choice_num <= len(options):
                            parameters['reasoning_effort'] = options[choice_num - 1]
                            print(f"Selected: {parameters['reasoning_effort']}")
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(options)}")
                    except ValueError:
                        print("Please enter a valid number or press Enter for default")
        
        # Configure task settings for local providers (LM-Studio, vLLM)
        if provider in ["lm-studio", "vllm"]:
            print("\n" + "="*40)
            print("ADVANCED TASK SETTINGS")
            print("="*40)
            print("\nThese settings are available for local model providers:")
            
        
        return parameters
    
    def choose_model(self, provider: str) -> Tuple[str, Dict[str, any]]:
        """
        Interactive model selection and parameter configuration.
        
        Args:
            provider: Selected provider name
            
        Returns:
            Tuple of (selected model name, configured parameters)
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
            reasoning_marker = " [Reasoning]" if model.get("is_reasoning") else ""
            description = f" - {model['description']}" if model.get("description") else ""
            print(f"{i}. {model['name']}{reasoning_marker}{description}")
        
        selected_model = None
        while True:
            try:
                choice = int(input(f"\nSelect model (1-{len(models)}): ").strip())
                if 1 <= choice <= len(models):
                    selected_model = models[choice - 1]
                    model_id = selected_model["id"]
                    print(f"Selected: {selected_model['name']}")
                    break
                else:
                    print(f"Please enter a number between 1 and {len(models)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                raise SystemExit(0)
        
        # Configure parameters for the selected model
        parameters = self.configure_model_parameters(provider, selected_model)
        
        # Show configuration summary
        if parameters:
            print("\n" + "="*40)
            print("CONFIGURATION SUMMARY")
            print("="*40)
            print(f"Model: {selected_model['name']}")
            for key, value in parameters.items():
                print(f"  - {key.replace('_', ' ').title()}: {value}")
        
        return model_id, parameters
    
    def get_model_info(self, provider: str, model_id: str) -> Optional[Dict]:
        """
        Get information about a specific model.
        
        Args:
            provider: Provider name
            model_id: Model ID
            
        Returns:
            Model information dictionary or None
        """
        models = self.config["models"].get(provider, [])
        for model in models:
            if model["id"] == model_id:
                return model
        return None