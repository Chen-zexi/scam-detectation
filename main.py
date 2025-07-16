#!/usr/bin/env python3
"""
Provides an interactive command-line interface for processing datasets.

This script offers a guided workflow for dataset processing with features like
configurable checkpointing and resume capabilities.

Key Features:
- Interactive prompts for all configuration options.
- Automatic discovery of datasets in the `data/` directory.
- Automatic detection of models from lm-studio and vLLM endpoints.
- Checkpoint listing, selection, and resumption.

Usage:
    python main.py

The script provides a guided workflow for:
1.  Choosing between annotation, evaluation, or transcript generation.
2.  Selecting a dataset from available options.
3.  Choosing a provider (e.g., OpenAI, LM-Studio, vLLM).
4.  Selecting a model.
5.  Configuring processing options.
6.  Resuming from checkpoints if available.
"""

import sys
import os
import asyncio
import requests
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

# Add src to path
sys.path.append('src')

from src.annotate import LLMAnnotationPipeline
from src.evaluate import ScamDetectionEvaluator
from src.synthesize import SynthesisGenerator, SynthesisPromptsManager


class InteractiveDatasetProcessor:
    """Manages the interactive command-line interface for dataset processing."""
    
    def __init__(self):
        self.config = {}
        self.available_datasets = []
        self.available_checkpoints = []
        
    def print_header(self):
        """Prints the application header."""
        print("="*80)
        print("INTERACTIVE DATASET PROCESSOR WITH CHECKPOINTING")
        print("="*80)
        print("This tool helps you process datasets for scam detection using various LLM providers.")
        print("You'll be guided through each step of the configuration process.")
        print()

    def discover_datasets(self) -> List[Dict[str, str]]:
        """Discovers available datasets in the `data/cleaned` directory."""
        print("Scanning for available datasets...")
        
        datasets = []
        data_dir = Path("data/cleaned")
        
        if not data_dir.exists():
            print("No data/ directory found!")
            return datasets
        
        # Look for CSV files in data/ subdirectories
        for csv_file in data_dir.rglob("*.csv"):
            try:
                # Quick check if it has required columns
                df_sample = pd.read_csv(csv_file, nrows=1)
                if 'label' in df_sample.columns:
                    # Get record count
                    total_records = len(pd.read_csv(csv_file))
                    
                    datasets.append({
                        'path': str(csv_file),
                        'name': csv_file.name,
                        'directory': str(csv_file.parent),
                        'columns': list(df_sample.columns),
                        'records': total_records
                    })
            except Exception as e:
                continue  # Skip invalid CSV files
        
        return datasets

    def choose_task(self) -> str:
        """Prompts the user to choose a task (annotation, evaluation, etc.)."""
        print("\nSTEP 1: Choose Task Type")
        print("-" * 40)
        print("1. Annotation - Generate structured annotations for datasets")
        print("2. Evaluation - Evaluate model performance on labeled datasets")
        print("3. Synthesis - Generate synthetic scam detection training data")
        
        while True:
            choice = input("\nSelect task type (1, 2, or 3): ").strip()
            if choice == "1":
                return "annotation"
            elif choice == "2":
                return "evaluation"
            elif choice == "3":
                return "synthesis"
            else:
                print("Please enter 1, 2, or 3")

    def choose_dataset(self) -> Dict[str, str]:
        """Prompts the user to select from available datasets."""
        print("\n STEP 2: Select Dataset")
        print("-" * 40)
        
        datasets = self.discover_datasets()
        
        if not datasets:
            print("No valid datasets found in data/ directory!")
            print("Please ensure your CSV files contain a 'label' column.")
            sys.exit(1)
        
        print(f"Found {len(datasets)} valid dataset(s):")
        print()
        
        for i, dataset in enumerate(datasets, 1):
            print(f"{i}. {dataset['name']}")
            print(f"Location: {dataset['directory']}")
            print(f"Records: {dataset['records']:,}")
            print(f"Columns: {', '.join(dataset['columns'])}")
            print()
        
        while True:
            try:
                choice = int(input(f"Select dataset (1-{len(datasets)}): ").strip())
                if 1 <= choice <= len(datasets):
                    selected = datasets[choice - 1]
                    print(f"Selected: {selected['name']}")
                    return selected
                else:
                    print(f"Please enter a number between 1 and {len(datasets)}")
            except ValueError:
                print("Please enter a valid number")

    def choose_provider(self) -> str:
        """Prompts the user to select an LLM provider."""
        print("\nSTEP 3: Select LLM Provider")
        print("-" * 40)
        print("1. OpenAI (GPT models)")
        print("2. LM Studio (Local models via LM Studio)")
        print("3. vLLM (High-performance inference server)")
        
        while True:
            choice = input("\nSelect provider (1, 2, or 3): ").strip()
            if choice == "1":
                return "openai"
            elif choice == "2":
                return "lm-studio"
            elif choice == "3":
                return "vllm"
            else:
                print("Please enter 1, 2, or 3")

    def get_openai_models(self) -> List[str]:
        """Returns a predefined list of OpenAI model options."""
        return [
            "o3",
            "gpt-4.1-mini", 
            "gpt-4.1"
        ]

    def get_lm_studio_models(self) -> List[str]:
        """Fetches available models from an LM Studio endpoint."""
        host_ip = input("Enter LM Studio host IP (default: localhost): ").strip() or "localhost"
        endpoint = f"http://{host_ip}:1234/v1/models"
        
        try:
            print(f"Connecting to LM Studio at {endpoint}...")
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = [model['id'] for model in data.get('data', [])]
            
            if not models:
                print("No models found on LM Studio server")
                return []
            
            print(f"Found {len(models)} model(s)")
            return models
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to LM Studio: {e}")
            print("Make sure LM Studio is running and accessible")
            return []

    def get_vllm_models(self) -> List[str]:
        """Fetches available models from vLLM endpoint."""
        host_ip = input("Enter vLLM host IP (default: host_ip configrued in .env): ").strip() or os.getenv("HOST_IP") or "localhost"
        endpoint = f"http://{host_ip}:8000/v1/models"
        
        try:
            print(f"Connecting to vLLM at {endpoint}...")
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = [model['id'] for model in data.get('data', [])]
            
            if not models:
                print("No models found on vLLM server")
                return []
            
            print(f"Found {len(models)} model(s)")
            return models
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to vLLM: {e}")
            print("Make sure vLLM server is running and accessible")
            return []

    def choose_synthesis_type(self) -> str:
        """Prompts the user to select a synthesis type."""
        print("\nSTEP 2a: Select Synthesis Type")
        print("-" * 40)
        
        prompts_manager = SynthesisPromptsManager()
        synthesis_types = prompts_manager.get_synthesis_types()
        
        print("Available synthesis types:")
        for i, syn_type in enumerate(synthesis_types, 1):
            type_info = prompts_manager.get_synthesis_type_info(syn_type)
            print(f"{i}. {type_info.get('name', syn_type)}")
            print(f"   {type_info.get('description', 'No description available')}")
            print()
        
        while True:
            try:
                choice = int(input(f"Select synthesis type (1-{len(synthesis_types)}): ").strip())
                if 1 <= choice <= len(synthesis_types):
                    selected = synthesis_types[choice - 1]
                    print(f"Selected: {prompts_manager.get_synthesis_type_info(selected).get('name', selected)}")
                    return selected
                else:
                    print(f"Please enter a number between 1 and {len(synthesis_types)}")
            except ValueError:
                print("Please enter a valid number")
    
    def choose_model(self, provider: str) -> str:
        """Prompts the user to select a model from the chosen provider."""
        print(f"\nSTEP 4: Select Model ({provider.upper()})")
        print("-" * 40)
        
        if provider == "openai":
            models = self.get_openai_models()
        elif provider == "lm-studio":
            models = self.get_lm_studio_models()
        elif provider == "vllm":
            models = self.get_vllm_models()
        else:
            print(f"Unknown provider: {provider}")
            return ""
        
        if not models:
            print("No models available. Please check your provider configuration.")
            sys.exit(1)
        
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

    def _infer_task_from_filename(self, filename: str) -> str:
        """Infers task type from a checkpoint filename for backward compatibility."""
        if 'transcript_generation' in filename or 'phone_transcript' in filename:
            return 'synthesis'
        elif 'annotation' in filename:
            return 'annotation'
        elif 'evaluation' in filename:
            return 'evaluation'
        elif any(syn_type in filename for syn_type in ['phishing_email', 'sms_scam']):
            return 'synthesis'
        else:
            return 'unknown'

    def discover_checkpoints(self) -> List[Dict[str, str]]:
        """Discovers available checkpoint files for the current task."""
        task = self.config.get('task', 'unknown')
        checkpoint_dir = Path("checkpoints") / task
        
        # Also check the old checkpoint directory for backward compatibility
        old_checkpoint_dir = Path("checkpoints")
        
        checkpoints = []
        
        # Check new task-specific directory first
        if checkpoint_dir.exists():
            for checkpoint_file in checkpoint_dir.glob("*.json"):
                try:
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    # Handle both total_records and total_target for compatibility
                    total = checkpoint_data.get('total_records', checkpoint_data.get('total_target', 0))
                    checkpoints.append({
                        'file': str(checkpoint_file),
                        'name': checkpoint_file.name,
                        'progress': f"{checkpoint_data.get('current_index', 0):,}/{total:,}",
                        'provider': checkpoint_data.get('provider', 'Unknown'),
                        'model': checkpoint_data.get('model', 'Unknown'),
                        'timestamp': checkpoint_data.get('timestamp', 'Unknown'),
                        'task': checkpoint_data.get('task', task)
                    })
                except:
                    continue  # Skip invalid checkpoint files
        
        # Check old directory for backward compatibility - only include matching task
        if old_checkpoint_dir.exists():
            for checkpoint_file in old_checkpoint_dir.glob("*.json"):
                try:
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    # Only include checkpoints for the current task
                    file_task = self._infer_task_from_filename(checkpoint_file.name)
                    if file_task == task:
                        # Handle both total_records and total_target for compatibility
                        total = checkpoint_data.get('total_records', checkpoint_data.get('total_target', 0))
                        checkpoints.append({
                            'file': str(checkpoint_file),
                            'name': checkpoint_file.name,
                            'progress': f"{checkpoint_data.get('current_index', 0):,}/{total:,}",
                            'provider': checkpoint_data.get('provider', 'Unknown'),
                            'model': checkpoint_data.get('model', 'Unknown'),
                            'timestamp': checkpoint_data.get('timestamp', 'Unknown'),
                            'task': checkpoint_data.get('task', file_task)
                        })
                except:
                    continue  # Skip invalid checkpoint files
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: Path(x['file']).stat().st_mtime, reverse=True)
        return checkpoints

    def choose_checkpoint(self) -> Optional[str]:
        """Prompts the user to select a checkpoint or start fresh."""
        print("\nSTEP 5: Checkpoint Options")
        print("-" * 40)
        
        checkpoints = self.discover_checkpoints()
        
        if not checkpoints:
            print("No existing checkpoints found. Starting fresh.")
            return None
        
        print(f"Found {len(checkpoints)} checkpoint(s):")
        print("0. Start fresh (no checkpoint)")
        
        for i, checkpoint in enumerate(checkpoints, 1):
            print(f"{i}. {checkpoint['name']}")
            print(f"    Progress: {checkpoint['progress']}")
            print(f"    Provider: {checkpoint['provider']}")
            print(f"    Model: {checkpoint['model']}")
            print(f"    Task: {checkpoint['task']}")
            print(f"    Time: {checkpoint['timestamp']}")
            print()
        
        while True:
            try:
                choice = int(input(f"Select option (0-{len(checkpoints)}): ").strip())
                if choice == 0:
                    print("Starting fresh without checkpoint")
                    return None
                elif 1 <= choice <= len(checkpoints):
                    selected = checkpoints[choice - 1]
                    print(f"Will resume from: {selected['name']}")
                    return selected['file']
                else:
                    print(f"Please enter a number between 0 and {len(checkpoints)}")
            except ValueError:
                print("Please enter a valid number")

    def configure_processing_options(self) -> Dict:
        """Configures advanced processing options based on user input."""
        print("\nSTEP 6: Processing Configuration")
        print("-" * 40)
        
        options = {}
        
        # Sample size configuration (only for annotation and evaluation)
        if self.config['task'] in ['annotation', 'evaluation']:
            print("\nSample Size Configuration:")
            total_records = self.config['dataset']['records']
            print(f"Total records in dataset: {total_records:,}")
            
            sample_all = input(f"Process all {total_records:,} records? (Y/n): ").strip().lower()
            if sample_all in ['n', 'no']:
                while True:
                    try:
                        sample_input = input(f"Enter sample size (1-{total_records:,}): ").strip()
                        sample_size = int(sample_input)
                        if 1 <= sample_size <= total_records:
                            options['sample_size'] = sample_size
                            break
                        else:
                            print(f"Please enter a number between 1 and {total_records:,}")
                    except ValueError:
                        print("Please enter a valid number")
                
                # Balanced sampling option
                balanced_choice = input("Use balanced sampling (equal scam/legitimate)? (y/N): ").strip().lower()
                options['balanced_sample'] = balanced_choice in ['y', 'yes']
                
                if options['balanced_sample']:
                    print("Note: Actual sample size may be smaller if dataset has insufficient samples of either class")
            else:
                options['sample_size'] = total_records
                options['balanced_sample'] = False
        else:
            # For synthesis, use the configured number of items
            options['sample_size'] = self.config['dataset']['records']
            options['balanced_sample'] = False
            if self.config['task'] == 'synthesis':
                type_info = SynthesisPromptsManager().get_synthesis_type_info(self.config.get('synthesis_type', 'phone_transcript'))
                print(f"\nSynthesis Generation: Will generate {options['sample_size']:,} {type_info.get('name', 'items')}")
        
        # Checkpoint interval
        default_interval = min(1000, max(1, options['sample_size'] // 10)) if options['sample_size'] < 10000 else 1000
        interval_input = input(f"Checkpoint interval (default: {default_interval}): ").strip()
        checkpoint_interval = int(interval_input) if interval_input else default_interval
        options['checkpoint_interval'] = max(1, checkpoint_interval)  # Ensure it's at least 1
        
        # Async processing
        if self.config['task'] == 'synthesis':
            # Synthesis is always async
            options['use_async'] = True
            print("\nSynthesis generation uses async processing by default.")
            default_concurrent = 5  # More conservative default for generation
            concurrent_input = input(f"Number of concurrent requests (default: {default_concurrent}): ").strip()
            options['concurrent_requests'] = int(concurrent_input) if concurrent_input else default_concurrent
        else:
            async_choice = input("Use async processing for faster execution? (Y/n): ").strip().lower()
            options['use_async'] = async_choice not in ['n', 'no']
            
            # Always set concurrent_requests (default to 1 for sequential processing)
            default_concurrent = 20 if options['use_async'] else 1
            if options['use_async']:
                concurrent_input = input(f"Number of concurrent requests (default: {default_concurrent}): ").strip()
                options['concurrent_requests'] = int(concurrent_input) if concurrent_input else default_concurrent
            else:
                options['concurrent_requests'] = 1  # Sequential processing
        
        # Content columns (only for annotation and evaluation)
        if self.config['task'] in ['annotation', 'evaluation']:
            content_input = input("Content columns (comma-separated, default: auto-detect): ").strip()
            if content_input:
                options['content_columns'] = [col.strip() for col in content_input.split(',')]
            else:
                options['content_columns'] = None
        else:
            # Synthesis doesn't use content columns
            options['content_columns'] = None
        
        # Advanced options (vary by task)
        if self.config['task'] == 'synthesis':
            # Use default settings for synthesis (bypass advanced options)
            options['enable_thinking'] = False
            options['use_structure_model'] = False
            print("\nUsing default settings for synthesis generation (thinking tokens: disabled, structure model: disabled)")
            
        else:
            print("\nAdvanced options:")
            print("Warning: You should set these to N for most cases.")
            
            thinking_choice = input("Enable thinking tokens? (y/N): ").strip().lower()
            options['enable_thinking'] = thinking_choice in ['y', 'yes']
            
            structure_choice = input("Use structure model for parsing? (y/N): ").strip().lower()
            options['use_structure_model'] = structure_choice in ['y', 'yes']
            
            if options['use_structure_model']:
                print("Note: Will use OpenAI gpt-4.1-nano for structured output parsing")
        
        return options
    
    def validate_checkpoint_compatibility(self):
        """Validates checkpoint compatibility and offers override options."""
        if not self.config.get('checkpoint_file'):
            return  # No checkpoint selected, nothing to validate
            
        try:
            with open(self.config['checkpoint_file'], 'r') as f:
                checkpoint_data = json.load(f)
            
            checkpoint_provider = checkpoint_data.get('provider', '')
            checkpoint_model = checkpoint_data.get('model', '')
            checkpoint_task = checkpoint_data.get('task', '')
            
            current_provider = self.config['provider']
            current_model = self.config['model']
            current_task = self.config['task']
            
            mismatches = []
            
            if checkpoint_provider != current_provider:
                mismatches.append(f"Provider: checkpoint='{checkpoint_provider}' vs current='{current_provider}'")
            
            if checkpoint_model != current_model:
                mismatches.append(f"Model: checkpoint='{checkpoint_model}' vs current='{current_model}'")
                
            if checkpoint_task != current_task:
                mismatches.append(f"Task: checkpoint='{checkpoint_task}' vs current='{current_task}'")
            
            if mismatches:
                print(f"\n*** CHECKPOINT COMPATIBILITY WARNING ***")
                print("The selected checkpoint was created with different settings:")
                for mismatch in mismatches:
                    print(f"  - {mismatch}")
                print()
                print("This might cause issues, but you can override and continue anyway.")
                print("The checkpoint data will be used, but with your current provider/model settings.")
                print()
                
                override = input("Continue anyway and override the differences? (Y/n): ").strip().lower()
                if override not in ['', 'y', 'yes']:
                    print("Checkpoint cancelled. Please select a compatible checkpoint or start fresh.")
                    self.config['checkpoint_file'] = None
                else:
                    print("Continuing with override. The checkpoint progress will be used with your current settings.")
            
        except Exception as e:
            print(f"\nError reading checkpoint file: {e}")
            print("The checkpoint file might be corrupted or invalid.")
            
            continue_anyway = input("Continue without checkpoint? (Y/n): ").strip().lower()
            if continue_anyway in ['', 'y', 'yes']:
                self.config['checkpoint_file'] = None
            else:
                print("Processing cancelled.")
                sys.exit(1)

    def print_configuration_summary(self):
        """Prints a final summary of the chosen configuration."""
        print("\nCONFIGURATION SUMMARY")
        print("="*50)
        print(f"Task: {self.config['task']}")
        
        if self.config['task'] == 'synthesis':
            print(f"Synthesis type: {self.config.get('synthesis_type', 'unknown')}")
            print(f"Items to generate: {self.config['dataset']['records']:,}")
        else:
            print(f"Dataset: {self.config['dataset']['name']} ({self.config['dataset']['records']:,} total records)")
            
            # Sample size information (only for annotation and evaluation)
            sample_size = self.config.get('sample_size', self.config['dataset']['records'])
            if sample_size == self.config['dataset']['records']:
                print(f"Sample size: All records ({sample_size:,})")
            else:
                print(f"Sample size: {sample_size:,} records")
                if self.config.get('balanced_sample', False):
                    print(f"Sampling: Balanced (equal scam/legitimate)")
                else:
                    print(f"Sampling: Random")
        
        print(f"Provider: {self.config['provider']}")
        print(f"Model: {self.config['model']}")
        if self.config.get('checkpoint_file'):
            print(f"Resume from: {Path(self.config['checkpoint_file']).name}")
        else:
            print("Starting: Fresh (no checkpoint)")
        print(f"Checkpoint interval: {self.config['checkpoint_interval']:,} records")
        if self.config['use_async']:
            print(f"Processing: Async ({self.config['concurrent_requests']} concurrent, overlapping batches)")
        else:
            print("Processing: Sequential")
        if self.config['content_columns']:
            print(f"Content columns: {', '.join(self.config['content_columns'])}")
        print()

    async def run_processing(self):
        """Runs the processing task based on the current configuration."""
        print("STARTING PROCESSING...")
        print("-" * 40)
        
        total_records = self.config['dataset']['records']
        sample_size = self.config.get('sample_size', total_records)
        
        # Handle checkpoint file for specific resume
        resume_from_checkpoint = self.config.get('checkpoint_file') is not None
        override_compatibility = self.config.get('checkpoint_file') is not None  # If user selected specific checkpoint, they already confirmed override
        
        # If specific checkpoint file is selected, temporarily rename it to be the "latest"
        # so the auto-discovery mechanism will find it
        original_checkpoint_name = None
        temp_checkpoint_name = None
        if self.config.get('checkpoint_file'):
            import shutil
            original_checkpoint_path = Path(self.config['checkpoint_file'])
            dataset_name = Path(self.config['dataset']['path']).stem
            provider = self.config['provider']
            model = self.config['model'].replace("/", "-").replace("\\", "-").replace(":", "-")
            task = self.config['task']
            
            # Generate expected checkpoint name pattern
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_checkpoint_name = f"checkpoints/{dataset_name}_{task}_{provider}_{model}_{timestamp}.json"
            
            # Copy the selected checkpoint to temp name
            try:
                shutil.copy2(original_checkpoint_path, temp_checkpoint_name)
                print(f"Prepared checkpoint for resumption: {Path(temp_checkpoint_name).name}")
            except Exception as e:
                print(f"Warning: Could not prepare checkpoint: {e}")
                resume_from_checkpoint = False
        
        # Create processor based on task type
        try:
            # Use configured sample size and balanced sampling
            sample_size = self.config.get('sample_size', total_records)
            balanced_sample = self.config.get('balanced_sample', False)
            
            if self.config['task'] == "annotation":
                processor = LLMAnnotationPipeline(
                    dataset_path=self.config['dataset']['path'],
                    provider=self.config['provider'],
                    model=self.config['model'],
                    sample_size=sample_size,
                    balanced_sample=balanced_sample,
                    content_columns=self.config['content_columns'],
                    output_dir="results/annotation",
                    enable_thinking=self.config['enable_thinking'],
                    use_structure_model=self.config['use_structure_model']
                )
            elif self.config['task'] == "evaluation":
                processor = ScamDetectionEvaluator(
                    dataset_path=self.config['dataset']['path'],
                    provider=self.config['provider'],
                    model=self.config['model'],
                    sample_size=sample_size,
                    balanced_sample=balanced_sample,
                    content_columns=self.config['content_columns'],
                    enable_thinking=self.config['enable_thinking'],
                    use_structure_model=self.config['use_structure_model']
                )
            else:  # synthesis
                # Use generic SynthesisGenerator for all synthesis types
                processor = SynthesisGenerator(
                        synthesis_type=self.config['synthesis_type'],
                        sample_size=total_records,
                        output_dir="results/synthesis",
                        provider=self.config['provider'],
                        model=self.config['model'],
                        enable_thinking=self.config['enable_thinking'],
                        use_structure_model=self.config['use_structure_model']
                    )
            
            # Run processing
            start_time = time.time()
            
            # Handle synthesis separately as it's always async
            if self.config['task'] == "synthesis":
                # Synthesis is always async
                results = await processor.process_full_generation_with_checkpoints(
                    checkpoint_interval=self.config['checkpoint_interval'],
                    checkpoint_dir="checkpoints/synthesis",
                    resume_from_checkpoint=resume_from_checkpoint,
                    concurrent_requests=self.config['concurrent_requests'],
                    override_compatibility=override_compatibility
                )
            else:
                # Determine if we're processing full dataset or sample
                is_full_dataset = sample_size == total_records
                
                if is_full_dataset and (resume_from_checkpoint or self.config.get('checkpoint_file')):
                    # Full dataset with checkpointing
                    task_checkpoint_dir = f"checkpoints/{self.config['task']}"
                    if self.config['use_async']:
                        if self.config['task'] == "annotation":
                            results = await processor.process_full_dataset_with_checkpoints_async(
                                checkpoint_interval=self.config['checkpoint_interval'],
                                checkpoint_dir=task_checkpoint_dir,
                                resume_from_checkpoint=resume_from_checkpoint,
                                concurrent_requests=self.config['concurrent_requests'],
                                override_compatibility=override_compatibility
                            )
                        else:
                            results = await processor.process_full_dataset_with_checkpoints_async(
                                checkpoint_interval=self.config['checkpoint_interval'],
                                checkpoint_dir=task_checkpoint_dir,
                                resume_from_checkpoint=resume_from_checkpoint,
                                concurrent_requests=self.config['concurrent_requests']
                            )
                    else:
                        if self.config['task'] == "annotation":
                            results = processor.process_full_dataset_with_checkpoints(
                                checkpoint_interval=self.config['checkpoint_interval'],
                                checkpoint_dir=task_checkpoint_dir,
                                resume_from_checkpoint=resume_from_checkpoint,
                                override_compatibility=override_compatibility
                            )
                        else:
                            results = processor.process_full_dataset_with_checkpoints(
                                checkpoint_interval=self.config['checkpoint_interval'],
                                checkpoint_dir=task_checkpoint_dir,
                                resume_from_checkpoint=resume_from_checkpoint
                            )
                else:
                    # Sample processing (no checkpointing for samples)
                    print("\nProcessing sample dataset...")
                    if self.config['use_async']:
                        if self.config['task'] == "annotation":
                            results = await processor.run_full_annotation_async(
                                concurrent_requests=self.config['concurrent_requests']
                            )
                        else:
                            results = await processor.run_full_evaluation_async(
                                concurrent_requests=self.config['concurrent_requests']
                            )
                    else:
                        if self.config['task'] == "annotation":
                            results = processor.run_full_annotation()
                        else:
                            results = processor.run_full_evaluation()
            
            # Clean up temp checkpoint file if created
            if temp_checkpoint_name and Path(temp_checkpoint_name).exists():
                try:
                    Path(temp_checkpoint_name).unlink()
                    print(f"Cleaned up temporary checkpoint file")
                except:
                    pass  # Ignore cleanup errors
            
            # Print results
            if self.config['task'] == "synthesis":
                # For synthesis, results have a different structure
                if 'status' in results and results['status'] == 'completed':
                    print(f"\nSynthesis generation completed successfully!")
                    if 'results' in results:
                        save_results = results['results']
                        print(f"Generated: {save_results.get('success_count', 0)} items")
                        print(f"Errors: {save_results.get('error_count', 0)} items")
                        print(f"Results saved to: {save_results.get('detailed_results', 'Unknown')}")
                else:
                    print(f"\nSynthesis generation failed: {results.get('error', 'Unknown error')}")
            else:
                self.print_results_summary(results)
            
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user")
            print("Progress has been saved to checkpoint. You can resume later by running this script again.")
        except Exception as e:
            print(f"\n\nError during processing: {e}")
            print("Check checkpoint files to see progress saved so far.")

    def print_results_summary(self, results: Dict):
        """Prints a summary of the processing results."""
        # Handle different result formats (checkpoint vs non-checkpoint)
        if 'summary' in results:
            # Checkpoint format
            summary = results['summary']
            save_paths = results['save_paths']
        else:
            # Non-checkpoint format (from run_full_evaluation)
            metrics = results.get('metrics', {})
            save_paths = results.get('save_paths', {})
            dataset_info = results.get('dataset_info', {})
            
            # Convert metrics to summary format
            summary = {
                'total_records': len(results.get('results', [])),
                'successful_evaluations': metrics.get('successfully_processed', 0),
                'correct_predictions': metrics.get('correct_predictions', 0),
                'accuracy': metrics.get('accuracy', 0),
                'success_rate': 1.0 if metrics.get('successfully_processed', 0) > 0 else 0
            }
            
            # For annotations
            if self.config['task'] == 'annotation':
                successful = sum(1 for r in results.get('results', []) if r.get('annotation_success', True))
                usable = sum(1 for r in results.get('results', []) if r.get('usability', True))
                total = len(results.get('results', []))
                summary = {
                    'total_records': total,
                    'successful_annotations': successful,
                    'usable_annotations': usable,
                    'success_rate': successful / total if total > 0 else 0,
                    'usability_rate': usable / total if total > 0 else 0
                }
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        # Handle different save_paths formats
        if 'results_file' in save_paths:
            print(f"Results saved to: {save_paths['results_file']}")
        elif 'results_directory' in save_paths:
            print(f"Results saved to: {save_paths['results_directory']}")
            
        if save_paths.get('summary_file'):
            print(f"Summary saved to: {save_paths['summary_file']}")
        if save_paths.get('checkpoint_file'):
            print(f"Final checkpoint: {save_paths['checkpoint_file']}")
        
        print(f"\nSummary Statistics:")
        print(f"  Total records: {summary['total_records']:,}")
        
        if self.config['task'] == 'evaluation':
            print(f"  Successful evaluations: {summary['successful_evaluations']:,}")
            print(f"  Correct predictions: {summary['correct_predictions']:,}")
            print(f"  Accuracy: {summary['accuracy']:.2%}")
            print(f"  Success rate: {summary['success_rate']:.2%}")
        elif self.config['task'] == 'annotation':
            print(f"  Successful annotations: {summary.get('successful_annotations', 0):,}")
            if 'usable_annotations' in summary:
                print(f"  Usable annotations: {summary['usable_annotations']:,}")
            print(f"  Success rate: {summary.get('success_rate', 0):.2%}")
            if 'usability_rate' in summary:
                print(f"  Usability rate: {summary['usability_rate']:.2%}")
        else:  # synthesis
            print(f"  Successful generations: {summary.get('success_count', 0):,}")
            print(f"  Errors: {summary.get('error_count', 0):,}")
            print(f"  Success rate: {summary.get('success_rate', 0):.2%}")
            if 'category_distribution' in summary:
                print(f"  Category distribution: {summary['category_distribution']}")

    async def run(self):
        """Runs the main interactive workflow."""
        self.print_header()
        
        # Step-by-step configuration
        self.config['task'] = self.choose_task()
        
        # For synthesis, we need to select the synthesis type
        if self.config['task'] == "synthesis":
            self.config['synthesis_type'] = self.choose_synthesis_type()
            
            # Get sample size for synthesis
            print("\nSTEP 2b: Configure Synthesis Generation")
            print("-" * 40)
            while True:
                try:
                    sample_size = int(input("Enter number of items to generate (default: 100): ").strip() or "100")
                    if sample_size > 0:
                        break
                    else:
                        print("Please enter a positive number")
                except ValueError:
                    print("Please enter a valid number")
            
            # Get synthesis type info
            prompts_manager = SynthesisPromptsManager()
            type_info = prompts_manager.get_synthesis_type_info(self.config['synthesis_type'])
            
            self.config['dataset'] = {
                'path': f'generated_{self.config["synthesis_type"]}',
                'name': type_info.get('name', 'Generated Data'),
                'directory': 'results/synthesis',
                'columns': list(prompts_manager.get_response_schema(self.config['synthesis_type']).keys()),
                'records': sample_size
            }
        else:
            self.config['dataset'] = self.choose_dataset()
            
        self.config['provider'] = self.choose_provider()
        self.config['model'] = self.choose_model(self.config['provider'])
        
        # Configure processing options (includes sample size)
        processing_options = self.configure_processing_options()
        self.config.update(processing_options)
        
        # Only offer checkpoint options if processing full dataset (or synthesis)
        if (self.config['task'] == 'synthesis' or 
            self.config.get('sample_size', self.config['dataset']['records']) == self.config['dataset']['records']):
            self.config['checkpoint_file'] = self.choose_checkpoint()
        else:
            self.config['checkpoint_file'] = None
            print("\nNote: Checkpointing is only available when processing the full dataset.")
        
        # Validate checkpoint compatibility if resuming
        self.validate_checkpoint_compatibility()
        
        # Show configuration summary and confirm
        self.print_configuration_summary()
        
        confirm = input("Start processing with this configuration? (Y/n): ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            await self.run_processing()
        else:
            print("Processing cancelled.")


def main():
    """Main entry point for the interactive dataset processor."""
    try:
        processor = InteractiveDatasetProcessor()
        asyncio.run(processor.run())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 