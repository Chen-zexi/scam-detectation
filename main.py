#!/usr/bin/env python3
"""
Refactored main entry point for the Interactive Dataset Processor.

This version uses modular CLI components for better separation of concerns.
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append('src')

from src.annotate import LLMAnnotationPipeline
from src.evaluate import ScamDetectionEvaluator
from src.synthesize import SynthesisGenerator
from src.cli import (
    DatasetManager,
    ModelSelector,
    CheckpointManager,
    ConfigManager,
    UIHelper
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable httpx logging to avoid cluttering the output
logging.getLogger("httpx").setLevel(logging.WARNING)


class InteractiveProcessor:
    """Main processor that orchestrates the CLI workflow."""
    
    def __init__(self):
        """Initialize all managers."""
        self.dataset_manager = DatasetManager()
        self.model_selector = ModelSelector()
        self.checkpoint_manager = CheckpointManager()
        self.config_manager = ConfigManager()
        self.ui_helper = UIHelper()
        self.config = {}
        
    async def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation task."""
        # Extract model parameters to pass to evaluator
        model_params = {}
        if 'reasoning_effort' in self.config:
            model_params['reasoning_effort'] = self.config['reasoning_effort']
        if 'verbosity' in self.config:
            model_params['verbosity'] = self.config['verbosity']
        
        evaluator = ScamDetectionEvaluator(
            dataset_path=self.config['dataset_path'],
            provider=self.config['provider'],
            model=self.config['model'],
            sample_size=self.config.get('sample_size'),
            balanced_sample=self.config.get('balanced_sample', True),
            enable_thinking=self.config.get('enable_thinking', False),
            use_structure_model=self.config.get('use_structure_model', False),
            **model_params
        )
        
        # Check if we're processing a sample or full dataset
        is_sample = self.config.get('sample_size') is not None
        
        if is_sample:
            # For samples, use the run_full_evaluation methods
            print("\nProcessing sample dataset...")
            if self.config.get('use_async', True):
                return await evaluator.run_full_evaluation_async(
                    concurrent_requests=self.config.get('concurrent_requests', 10)
                )
            else:
                return evaluator.run_full_evaluation()
        else:
            # For full dataset, use checkpoint methods
            if self.config.get('use_async', True):
                return await evaluator.process_full_dataset_with_checkpoints_async(
                    checkpoint_interval=self.config.get('checkpoint_interval', 100),
                    resume_from_checkpoint=bool(self.config.get('checkpoint_file')),
                    concurrent_requests=self.config.get('concurrent_requests', 10)
                )
            else:
                return evaluator.process_full_dataset_with_checkpoints(
                    checkpoint_interval=self.config.get('checkpoint_interval', 100),
                    resume_from_checkpoint=bool(self.config.get('checkpoint_file'))
                )
    
    async def run_annotation(self) -> Dict[str, Any]:
        """Run annotation task."""
        # Extract model parameters to pass to annotator
        model_params = {}
        if 'reasoning_effort' in self.config:
            model_params['reasoning_effort'] = self.config['reasoning_effort']
        if 'verbosity' in self.config:
            model_params['verbosity'] = self.config['verbosity']
        
        annotator = LLMAnnotationPipeline(
            dataset_path=self.config['dataset_path'],
            provider=self.config['provider'],
            model=self.config['model'],
            sample_size=self.config.get('sample_size'),
            balanced_sample=self.config.get('balanced_sample', True),
            enable_thinking=self.config.get('enable_thinking', False),
            use_structure_model=self.config.get('use_structure_model', False),
            **model_params
        )
        
        # Check if we're processing a sample or full dataset
        is_sample = self.config.get('sample_size') is not None
        
        if is_sample:
            # For samples, use the run_full_annotation methods
            print("\nProcessing sample dataset...")
            if self.config.get('use_async', True):
                return await annotator.run_full_annotation_async(
                    concurrent_requests=self.config.get('concurrent_requests', 10)
                )
            else:
                return annotator.run_full_annotation()
        else:
            # For full dataset, use checkpoint methods
            if self.config.get('use_async', True):
                return await annotator.process_full_dataset_with_checkpoints_async(
                    checkpoint_interval=self.config.get('checkpoint_interval', 100),
                    resume_from_checkpoint=bool(self.config.get('checkpoint_file')),
                    concurrent_requests=self.config.get('concurrent_requests', 10)
                )
            else:
                return annotator.process_full_dataset_with_checkpoints(
                    checkpoint_interval=self.config.get('checkpoint_interval', 100),
                    resume_from_checkpoint=bool(self.config.get('checkpoint_file'))
                )
    
    async def run_synthesis(self) -> Dict[str, Any]:
        """Run synthesis task."""
        # Extract model parameters to pass to generator
        model_params = {}
        if 'reasoning_effort' in self.config:
            model_params['reasoning_effort'] = self.config['reasoning_effort']
        if 'verbosity' in self.config:
            model_params['verbosity'] = self.config['verbosity']
        
        generator = SynthesisGenerator(
            synthesis_type=self.config['synthesis_type'],
            sample_size=self.config.get('sample_size', 100),
            provider=self.config['provider'],
            model=self.config['model'],
            enable_thinking=self.config.get('enable_thinking', False),
            use_structure_model=self.config.get('use_structure_model', False),
            save_to_mongodb=self.config.get('save_to_mongodb', True),
            category=self.config.get('category', 'ALL'),
            **model_params
        )
        
        return await generator.process_full_generation_with_checkpoints(
            checkpoint_interval=self.config.get('checkpoint_interval', 100),
            resume_from_checkpoint=bool(self.config.get('checkpoint_file')),
            concurrent_requests=self.config.get('concurrent_requests', 5)
        )
    
    async def run(self):
        """Main execution flow."""
        # Print header
        self.ui_helper.print_header()
        
        # Step 1: Choose task
        task = self.ui_helper.choose_task()
        self.config['task'] = task
        
        # Step 2: Task-specific configuration
        if task == "synthesis":
            # Choose synthesis type
            synthesis_type = self.ui_helper.choose_synthesis_type()
            self.config['synthesis_type'] = synthesis_type
            
            # Choose category
            category = self.ui_helper.choose_synthesis_category(synthesis_type)
            self.config['category'] = category
        else:
            # Choose dataset
            datasets = self.dataset_manager.discover_datasets()
            if not datasets:
                print("\nNo datasets found in data/cleaned/")
                print("Please add CSV datasets with a 'label' column.")
                return
            
            dataset = self.dataset_manager.choose_dataset(datasets)
            self.config['dataset_path'] = dataset['path']
            self.config['dataset_name'] = dataset['name']
            
            # Step 2a: Ask if user wants to check for checkpoints (for evaluation/annotation)
            if self.checkpoint_manager.ask_checkpoint_preference():
                # Check for existing checkpoints for this dataset
                checkpoints = self.checkpoint_manager.discover_checkpoints(
                    dataset_name=self.config['dataset_name'],
                    task=task,
                    incomplete_only=True
                )
                
                if checkpoints:
                    checkpoint_file = self.checkpoint_manager.choose_checkpoint(checkpoints)
                    if checkpoint_file:
                        self.config['checkpoint_file'] = checkpoint_file
                        self.config['resume_from_checkpoint'] = True
                        
                        # Load configuration from checkpoint
                        checkpoint_config = self.checkpoint_manager.load_checkpoint_config(checkpoint_file)
                        if checkpoint_config:
                            # Store the original values that we want to keep
                            dataset_path = self.config['dataset_path']
                            dataset_name = self.config['dataset_name']
                            task = self.config['task']
                            
                            # Update config with checkpoint values
                            self.config.update(checkpoint_config)
                            
                            # Restore the values we want to keep
                            self.config['dataset_path'] = dataset_path
                            self.config['dataset_name'] = dataset_name
                            self.config['task'] = task
                            
                            print("\nCheckpoint selected. Configuration loaded:")
                            print(f"  Provider: {self.config['provider']}")
                            print(f"  Model: {self.config['model']}")
                            print(f"  Progress: {checkpoint_config['checkpoint_info']['current_index']}/{checkpoint_config['checkpoint_info']['total_records']} ({checkpoint_config['checkpoint_info']['progress']:.1%})")
                            
                            # Skip provider and model selection since we have them from checkpoint
                            self.config['skip_provider_selection'] = True
                            self.config['skip_model_selection'] = True
                else:
                    print("\nNo incomplete checkpoints found for this dataset and task.")
                    self.config['checkpoint_file'] = None
                    self.config['resume_from_checkpoint'] = False
            else:
                self.config['checkpoint_file'] = None
                self.config['resume_from_checkpoint'] = False
        
        # Step 3: Choose provider (skip if loaded from checkpoint)
        if not self.config.get('skip_provider_selection', False):
            provider = self.model_selector.choose_provider()
            self.config['provider'] = provider
        else:
            provider = self.config['provider']
            print(f"\nUsing provider from checkpoint: {provider}")
        
        # Step 4: Choose model (skip if loaded from checkpoint)
        if not self.config.get('skip_model_selection', False):
            model, model_parameters = self.model_selector.choose_model(provider)
            self.config['model'] = model
            # Store model parameters in config
            self.config.update(model_parameters)
        else:
            model = self.config['model']
            print(f"Using model from checkpoint: {model}")
        
        # Step 5: Configure processing options (skip if loaded from checkpoint)
        dataset_path = self.config.get('dataset_path', '')
        if not self.config.get('resume_from_checkpoint', False):
            processing_options = self.config_manager.configure_processing_options(task, dataset_path)
            self.config.update(processing_options)
        else:
            print("\nUsing processing options from checkpoint")
        
        # Step 6: Handle checkpoints for synthesis or validate existing checkpoint
        if task == 'synthesis':
            # For synthesis, always check for checkpoints after configuration
            if self.checkpoint_manager.ask_checkpoint_preference():
                checkpoints = self.checkpoint_manager.discover_checkpoints(
                    task=task,
                    provider=provider,
                    model=model,
                    incomplete_only=True
                )
                
                if checkpoints:
                    checkpoint_file = self.checkpoint_manager.choose_checkpoint(checkpoints)
                    if checkpoint_file:
                        self.config['checkpoint_file'] = checkpoint_file
                        self.config['resume_from_checkpoint'] = True
                        
                        # Load configuration from checkpoint for synthesis
                        checkpoint_config = self.checkpoint_manager.load_checkpoint_config(checkpoint_file)
                        if checkpoint_config:
                            self.config.update(checkpoint_config)
                            print("\nCheckpoint selected. Configuration loaded.")
                            print(f"  Progress: {checkpoint_config['checkpoint_info']['current_index']}/{checkpoint_config['checkpoint_info']['total_records']} ({checkpoint_config['checkpoint_info']['progress']:.1%})")
                    else:
                        self.config['resume_from_checkpoint'] = False
                else:
                    print("\nNo incomplete checkpoints found for synthesis.")
        elif self.config.get('checkpoint_file'):
            # For evaluation/annotation, validate the previously selected checkpoint
            if not self.checkpoint_manager.validate_checkpoint_compatibility(
                self.config['checkpoint_file'],
                self.config.get('dataset_name', ''),
                task,
                provider,
                model
            ):
                print("\nCheckpoint validation failed. Starting fresh.")
                self.config.pop('checkpoint_file', None)
                self.config['resume_from_checkpoint'] = False
        
        # Only show checkpoint note if not using checkpoints and sample size > 1000
        sample_size = self.config.get('sample_size')
        if not self.config.get('checkpoint_file') and sample_size and sample_size > 1000:
            print("\nNote: Checkpoint saving will be enabled for this large dataset.")
        
        # Print configuration summary
        self.config_manager.print_configuration_summary(self.config)
        
        # Confirm to proceed
        response = input("\nProceed with processing? (y/n) [y]: ").strip().lower()
        if response == 'n':
            print("Operation cancelled")
            return
        
        # Run processing
        print("\nStarting processing...")
        start_time = asyncio.get_event_loop().time()
        
        try:
            if task == "evaluation":
                results = await self.run_evaluation()
            elif task == "annotation":
                results = await self.run_annotation()
            elif task == "synthesis":
                results = await self.run_synthesis()
            else:
                raise ValueError(f"Unknown task: {task}")
            
            # Calculate elapsed time
            elapsed_time = asyncio.get_event_loop().time() - start_time
            print(f"\nProcessing completed in {elapsed_time:.1f} seconds")
            
            # Print results summary
            self.ui_helper.print_results_summary(results, task)
            
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user")
            print("Progress has been saved to checkpoint (if enabled)")
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            print(f"\nError during processing: {e}")
            print("Check the logs for more details")


def main():
    """Main entry point."""
    processor = InteractiveProcessor()
    
    try:
        asyncio.run(processor.run())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()