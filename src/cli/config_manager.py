"""
Configuration management for processing options.
"""

import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages processing configuration options."""
    
    def configure_processing_options(self, task: str, dataset_path: str) -> Dict[str, Any]:
        """
        Interactive configuration of processing options.
        
        Args:
            task: Task type (evaluation, annotation, synthesis)
            dataset_path: Path to the dataset
            
        Returns:
            Configuration dictionary
        """
        print("\nSTEP 6: Configure Processing Options")
        print("-" * 40)
        
        config = {}
        
        # Sample size configuration
        if task != "synthesis":
            dataset_name = Path(dataset_path).name
            print(f"\nDataset: {dataset_name}")
            print("\nHow many records to process?")
            print("1. All records")
            print("2. Custom sample size")
            
            while True:
                try:
                    choice = int(input("\nSelect option (1-2): ").strip())
                    if choice == 1:
                        config['sample_size'] = None
                        print("Processing all records")
                        break
                    elif choice == 2:
                        size = int(input("Enter sample size: ").strip())
                        if size > 0:
                            config['sample_size'] = size
                            print(f"Processing {size} records")
                            break
                        else:
                            print("Sample size must be positive")
                    else:
                        print("Please enter 1 or 2")
                except ValueError:
                    print("Please enter a valid number")
        else:
            # For synthesis, always ask for sample size
            while True:
                try:
                    size = int(input("\nHow many items to generate: ").strip())
                    if size > 0:
                        config['sample_size'] = size
                        break
                    else:
                        print("Please enter a positive number")
                except ValueError:
                    print("Please enter a valid number")
        
        # Balanced sampling (for evaluation/annotation)
        if task in ["evaluation", "annotation"]:
            print("\nUse balanced sampling?")
            print("(Equal number of positive and negative samples)")
            
            response = input("Enable balanced sampling? (y/n) [y]: ").strip().lower()
            config['balanced_sample'] = response != 'n'
            print(f"Balanced sampling: {'Enabled' if config['balanced_sample'] else 'Disabled'}")
        
        # Concurrent requests (async is always enabled now)
        print("\nConcurrent Requests:")
        print("Number of parallel API calls (higher = faster but may hit rate limits)")
        
        while True:
            try:
                concurrent = input("Concurrent requests (1-20) [10]: ").strip()
                if not concurrent:
                    config['concurrent_requests'] = 10
                    break
                else:
                    concurrent = int(concurrent)
                    if 1 <= concurrent <= 20:
                        config['concurrent_requests'] = concurrent
                        break
                    else:
                        print("Please enter a number between 1 and 20")
            except ValueError:
                print("Please enter a valid number")
        
        print(f"Concurrent requests: {config['concurrent_requests']}")
        
        # Advanced options are configured in model_selector for local providers
        # Set defaults here - they will be overridden if configured in model_selector
        
        # MongoDB saving (for synthesis)
        if task == "synthesis":
            response = input("Save to MongoDB? (y/n) [y]: ").strip().lower()
            config['save_to_mongodb'] = response != 'n'
            print(f"MongoDB saving: {'Enabled' if config['save_to_mongodb'] else 'Disabled'}")
        
        # Checkpoint interval
        print("\nCheckpointing:")
        print("Save progress periodically to allow resuming")
        
        while True:
            try:
                interval = input("Checkpoint interval (records) [100]: ").strip()
                if not interval:
                    config['checkpoint_interval'] = 100
                    break
                else:
                    interval = int(interval)
                    if interval > 0:
                        config['checkpoint_interval'] = interval
                        break
                    else:
                        print("Checkpoint interval must be positive")
            except ValueError:
                print("Please enter a valid number")
        
        print(f"Checkpoint interval: {config['checkpoint_interval']} records")
        
        return config
    
    def print_configuration_summary(self, config: Dict[str, Any]):
        """
        Print a summary of the configuration.
        
        Args:
            config: Complete configuration dictionary
        """
        print("\n" + "="*80)
        print("CONFIGURATION SUMMARY")
        print("="*80)
        
        # Basic settings
        print(f"\nTask: {config.get('task', 'Unknown').upper()}")
        
        if 'dataset_name' in config:
            print(f"Dataset: {config['dataset_name']}")
            print(f"Dataset Path: {config['dataset_path']}")
        
        if 'synthesis_type' in config:
            print(f"Synthesis Type: {config['synthesis_type']}")
            if 'category' in config:
                print(f"Category: {config['category']}")
        
        print(f"Provider: {config.get('provider', 'Unknown')}")
        print(f"Model: {config.get('model', 'Unknown')}")
        
        # Processing options
        print("\nProcessing Options:")
        
        if config.get('sample_size'):
            print(f"  Sample Size: {config['sample_size']:,}")
        else:
            print(f"  Sample Size: All records")
        
        if 'balanced_sample' in config:
            print(f"  Balanced Sampling: {config['balanced_sample']}")
        
        print(f"  Concurrent Requests: {config.get('concurrent_requests', 10)}")
        
        
        if 'save_to_mongodb' in config:
            print(f"  Save to MongoDB: {config['save_to_mongodb']}")
        
        print(f"  Checkpoint Interval: {config.get('checkpoint_interval', 100)}")
        
        # Checkpoint info
        if config.get('checkpoint_file'):
            print(f"\nResuming from checkpoint: {Path(config['checkpoint_file']).name}")
        
        print("\n" + "="*80)