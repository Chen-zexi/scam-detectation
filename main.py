#!/usr/bin/env python3
"""
Interactive Full Dataset Processor with Checkpointing

This script provides an interactive command-line interface for processing
entire datasets with configurable checkpointing and resume capabilities.

Features:
- Interactive prompts for all configuration options
- Automatic dataset detection in data/ directory
- Automatic model detection for lm-studio and vllm endpoints
- Checkpoint listing and selection
- Resume from specific checkpoints

Usage:
    python main.py

The script will guide you through:
1. Choosing between annotation or evaluation
2. Selecting a dataset from available options
3. Choosing a provider (OpenAI, LM-Studio, vLLM)
4. Selecting a model
5. Configuring processing options
6. Resume from checkpoints if available
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

from src.annotation_pipeline import LLMAnnotationPipeline
from src.evaluator import ScamDetectionEvaluator


class TimeEstimator:
    """Helper class for tracking and estimating processing time"""
    
    def __init__(self, total_records: int, initial_estimate: float = 2.0):
        self.total_records = total_records
        self.initial_estimate = initial_estimate
        self.start_time = None
        self.last_update_time = None
        self.processed_count = 0
        
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
    def update(self, processed_count: int):
        """Update with current progress"""
        self.processed_count = processed_count
        self.last_update_time = time.time()
        
    def get_current_rate(self) -> float:
        """Get current processing rate (records per second)"""
        if not self.start_time or self.processed_count == 0:
            return 1.0 / self.initial_estimate
            
        elapsed = time.time() - self.start_time
        return self.processed_count / elapsed if elapsed > 0 else 1.0 / self.initial_estimate
        
    def get_estimated_remaining_time(self) -> float:
        """Get estimated remaining time in seconds"""
        remaining_records = self.total_records - self.processed_count
        rate = self.get_current_rate()
        return remaining_records / rate if rate > 0 else 0
        
    def format_time(self, seconds: float) -> str:
        """Format time in a human-readable way"""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds/3600:.1f} hours"


class InteractiveDatasetProcessor:
    """Interactive command-line interface for dataset processing"""
    
    def __init__(self):
        self.config = {}
        self.available_datasets = []
        self.available_checkpoints = []
        
    def print_header(self):
        """Print the application header"""
        print("="*80)
        print("INTERACTIVE DATASET PROCESSOR WITH CHECKPOINTING")
        print("="*80)
        print("This tool helps you process datasets for scam detection using various LLM providers.")
        print("You'll be guided through each step of the configuration process.")
        print()

    def discover_datasets(self) -> List[Dict[str, str]]:
        """Discover available datasets in data/ directory"""
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
        """Let user choose between annotation and evaluation"""
        print("\nSTEP 1: Choose Task Type")
        print("-" * 40)
        print("1. Annotation - Generate structured annotations for datasets")
        print("2. Evaluation - Evaluate model performance on labeled datasets")
        
        while True:
            choice = input("\nSelect task type (1 or 2): ").strip()
            if choice == "1":
                return "annotation"
            elif choice == "2":
                return "evaluation"
            else:
                print("Please enter 1 or 2")

    def choose_dataset(self) -> Dict[str, str]:
        """Let user choose from available datasets"""
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
        """Let user choose from available providers"""
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
        """Get predefined OpenAI model options"""
        return [
            "o3",
            "gpt-4.1-mini", 
            "gpt-4.1"
        ]

    def get_lm_studio_models(self) -> List[str]:
        """Get available models from LM Studio endpoint"""
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
        """Get available models from vLLM endpoint"""
        host_ip = input("Enter vLLM host IP (default: host_ip configrued in .env): ").strip() or "localhost"
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

    def choose_model(self, provider: str) -> str:
        """Let user choose from available models based on provider"""
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

    def discover_checkpoints(self) -> List[Dict[str, str]]:
        """Discover available checkpoint files"""
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return []
        
        checkpoints = []
        for checkpoint_file in checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                
                checkpoints.append({
                    'file': str(checkpoint_file),
                    'name': checkpoint_file.name,
                    'progress': f"{checkpoint_data.get('current_index', 0):,}/{checkpoint_data.get('total_records', 0):,}",
                    'provider': checkpoint_data.get('provider', 'Unknown'),
                    'model': checkpoint_data.get('model', 'Unknown'),
                    'timestamp': checkpoint_data.get('timestamp', 'Unknown'),
                    'task': checkpoint_data.get('task', 'Unknown')
                })
            except:
                continue  # Skip invalid checkpoint files
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: Path(x['file']).stat().st_mtime, reverse=True)
        return checkpoints

    def choose_checkpoint(self) -> Optional[str]:
        """Let user choose to resume from a checkpoint"""
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
        """Configure processing options"""
        print("\nSTEP 6: Processing Configuration")
        print("-" * 40)
        
        options = {}
        
        # Checkpoint interval
        default_interval = 1000
        interval_input = input(f"Checkpoint interval (default: {default_interval}): ").strip()
        options['checkpoint_interval'] = int(interval_input) if interval_input else default_interval
        
        # Async processing
        async_choice = input("Use async processing for faster execution? (Y/n): ").strip().lower()
        options['use_async'] = async_choice not in ['n', 'no']
        
        if options['use_async']:
            # Concurrent requests
            default_concurrent = 20
            concurrent_input = input(f"Number of concurrent requests (default: {default_concurrent}): ").strip()
            options['concurrent_requests'] = int(concurrent_input) if concurrent_input else default_concurrent
        
        # Content columns
        content_input = input("Content columns (comma-separated, default: auto-detect): ").strip()
        if content_input:
            options['content_columns'] = [col.strip() for col in content_input.split(',')]
        else:
            options['content_columns'] = None
        
        # Advanced options
        print("\nAdvanced options:")
        options['enable_thinking'] = input("Enable thinking tokens? (Y/n): ").strip().lower() not in ['n', 'no']
        options['use_structure_model'] = input("Use structure model for parsing? (Y/n): ").strip().lower() not in ['n', 'no']
        
        return options
    
    def validate_checkpoint_compatibility(self):
        """Validate checkpoint compatibility and offer override options"""
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
        """Print final configuration summary"""
        print("\nCONFIGURATION SUMMARY")
        print("="*50)
        print(f"Task: {self.config['task']}")
        print(f"Dataset: {self.config['dataset']['name']} ({self.config['dataset']['records']:,} records)")
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

    def _calculate_time_estimate(self, total_records: int) -> float:
        """Calculate initial time estimate per record using provider-specific baselines"""
        
        # Provider-specific base estimates (seconds per record)
        # These are conservative estimates that will be improved by real-time data during processing
        provider_estimates = {
            'openai': 1.5,      # OpenAI models are typically fast
            'lm-studio': 2.5,   # Local models vary more 
            'vllm': 1.0         # vLLM is optimized for throughput
        }
        
        # Get base estimate
        base_estimate = provider_estimates.get(self.config['provider'], 2.0)
        
        # Adjust for task complexity
        if self.config['task'] == 'annotation':
            base_estimate *= 1.2  # Annotations are slightly more complex
        
        print(f"Using conservative provider-based estimate: {base_estimate:.2f}s per record")
        print(f"   (Provider: {self.config['provider']}, Task: {self.config['task']})")
        print(f"   Note: This estimate will improve significantly once processing begins")
        return base_estimate
    

    
    def _display_time_estimate(self, total_est_time: float, remaining_records: int, time_per_record: float):
        """Display initial time estimate with context about real-time improvements"""
        
        print(f"\nINITIAL TIME ESTIMATION")
        print(f"   Records remaining: {remaining_records:,}")
        print(f"   Conservative estimate: {time_per_record:.2f}s per record")
        
        if self.config['use_async']:
            concurrent = self.config['concurrent_requests']
            print(f"   Concurrency: {concurrent} requests (overlapping batches)")
            
            # Show effective rate with overlapping efficiency
            effective_rate = concurrent * 1.3 / time_per_record  # Include efficiency factor
            print(f"   Theoretical max rate: ~{effective_rate:.1f} records/s (with overlapping efficiency)")
        
        # Display estimate in most appropriate unit
        if total_est_time > 3600:
            hours = total_est_time / 3600
            print(f"Conservative estimate: {hours:.1f} hours")
            if hours > 24:
                days = hours / 24
                print(f"      ({days:.1f} days)")
        elif total_est_time > 60:
            minutes = total_est_time / 60
            print(f"Conservative estimate: {minutes:.1f} minutes")
        else:
            print(f"Conservative estimate: {total_est_time:.0f} seconds")
        
        # Add helpful context about real-time updates
        print(f"\nNote: This is a conservative initial estimate based on provider averages.")
        print(f"Actual processing speed and time estimates will be calculated and displayed")
        print(f"in real-time based on current batch processing performance.")
        if self.config['provider'] == 'vllm':
            print(f"vLLM typically achieves much faster rates than this conservative estimate.")
        print()

    async def run_processing(self):
        """Run the actual processing based on configuration"""
        print("STARTING PROCESSING...")
        print("-" * 40)
        
        # Initial conservative time estimation (will be improved by real-time data during processing)
        total_records = self.config['dataset']['records']
        est_time_per_record = self._calculate_time_estimate(total_records)
        
        remaining_records = total_records
        if self.config.get('checkpoint_file'):
            try:
                with open(self.config['checkpoint_file'], 'r') as f:
                    checkpoint_data = json.load(f)
                    current_index = checkpoint_data.get('current_index', 0)
                    remaining_records = total_records - current_index
            except:
                pass
        
        # Calculate estimated time with concurrency factor
        if self.config['use_async']:
            # Account for concurrency and overlapping batches efficiency
            concurrency_factor = self.config['concurrent_requests']
            # Overlapping batches are ~20-40% more efficient
            efficiency_factor = 1.3
            total_est_time = (remaining_records * est_time_per_record) / (concurrency_factor * efficiency_factor)
        else:
            total_est_time = remaining_records * est_time_per_record
        
        # Display initial conservative estimate
        self._display_time_estimate(total_est_time, remaining_records, est_time_per_record)
        
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
            if self.config['task'] == "annotation":
                processor = LLMAnnotationPipeline(
                    dataset_path=self.config['dataset']['path'],
                    provider=self.config['provider'],
                    model=self.config['model'],
                    sample_size=total_records,
                    content_columns=self.config['content_columns'],
                    output_dir="results/full_dataset",
                    enable_thinking=self.config['enable_thinking'],
                    use_structure_model=self.config['use_structure_model']
                )
            else:  # evaluation
                processor = ScamDetectionEvaluator(
                    dataset_path=self.config['dataset']['path'],
                    provider=self.config['provider'],
                    model=self.config['model'],
                    sample_size=total_records,
                    content_columns=self.config['content_columns'],
                    enable_thinking=self.config['enable_thinking'],
                    use_structure_model=self.config['use_structure_model']
                )
            
            # Run processing
            start_time = time.time()
            if self.config['use_async']:
                if self.config['task'] == "annotation":
                    results = await processor.process_full_dataset_with_checkpoints_async(
                        checkpoint_interval=self.config['checkpoint_interval'],
                        checkpoint_dir="checkpoints",
                        resume_from_checkpoint=resume_from_checkpoint,
                        concurrent_requests=self.config['concurrent_requests'],
                        override_compatibility=override_compatibility
                    )
                else:
                    results = await processor.process_full_dataset_with_checkpoints_async(
                        checkpoint_interval=self.config['checkpoint_interval'],
                        checkpoint_dir="checkpoints",
                        resume_from_checkpoint=resume_from_checkpoint,
                        concurrent_requests=self.config['concurrent_requests']
                    )
            else:
                if self.config['task'] == "annotation":
                    results = processor.process_full_dataset_with_checkpoints(
                        checkpoint_interval=self.config['checkpoint_interval'],
                        checkpoint_dir="checkpoints",
                        resume_from_checkpoint=resume_from_checkpoint,
                        override_compatibility=override_compatibility
                    )
                else:
                    results = processor.process_full_dataset_with_checkpoints(
                        checkpoint_interval=self.config['checkpoint_interval'],
                        checkpoint_dir="checkpoints",
                        resume_from_checkpoint=resume_from_checkpoint
                    )
            
            # Clean up temp checkpoint file if created
            if temp_checkpoint_name and Path(temp_checkpoint_name).exists():
                try:
                    Path(temp_checkpoint_name).unlink()
                    print(f"Cleaned up temporary checkpoint file")
                except:
                    pass  # Ignore cleanup errors
            
            # Print results
            self.print_results_summary(results)
            
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user")
            print("Progress has been saved to checkpoint. You can resume later by running this script again.")
        except Exception as e:
            print(f"\n\nError during processing: {e}")
            print("Check checkpoint files to see progress saved so far.")

    def print_results_summary(self, results: Dict):
        """Print summary of processing results"""
        summary = results['summary']
        save_paths = results['save_paths']
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        print(f"Results saved to: {save_paths['results_file']}")
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
        else:  # annotation
            print(f"  Successful annotations: {summary['successful_annotations']:,}")
            print(f"  Usable annotations: {summary['usable_annotations']:,}")
            print(f"  Success rate: {summary['success_rate']:.2%}")
            print(f"  Usability rate: {summary['usability_rate']:.2%}")

    async def run(self):
        """Main interactive workflow"""
        self.print_header()
        
        # Step-by-step configuration
        self.config['task'] = self.choose_task()
        self.config['dataset'] = self.choose_dataset()
        self.config['provider'] = self.choose_provider()
        self.config['model'] = self.choose_model(self.config['provider'])
        self.config['checkpoint_file'] = self.choose_checkpoint()
        
        # Configure processing options
        processing_options = self.configure_processing_options()
        self.config.update(processing_options)
        
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
    """Main entry point"""
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