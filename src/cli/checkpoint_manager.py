"""
Checkpoint discovery and management functionality.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Handles checkpoint discovery and selection."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Root directory for checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        
    def discover_checkpoints(self, dataset_name: Optional[str] = None,
                           task: Optional[str] = None,
                           provider: Optional[str] = None,
                           model: Optional[str] = None,
                           incomplete_only: bool = True) -> List[Dict[str, str]]:
        """
        Discovers available checkpoint files.
        
        Args:
            dataset_name: Filter by dataset name
            task: Filter by task type
            provider: Filter by provider
            model: Filter by model
            incomplete_only: If True, only return incomplete checkpoints (progress < 1.0)
            
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        # Determine search directories based on task
        if task:
            # Look in task-specific directory
            task_checkpoint_dir = self.checkpoint_dir / task
            if task_checkpoint_dir.exists():
                search_dirs = [task_checkpoint_dir]
            else:
                search_dirs = []
        else:
            # Search all task subdirectories
            search_dirs = [d for d in self.checkpoint_dir.iterdir() if d.is_dir()]
            # Also include root for backward compatibility
            if self.checkpoint_dir.exists():
                search_dirs.append(self.checkpoint_dir)
        
        # Search in directories for JSON checkpoint files
        for search_dir in search_dirs:
            for checkpoint_file in search_dir.glob("*.json"):
                try:
                    with open(checkpoint_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract metadata
                    checkpoint_info = {
                        'path': str(checkpoint_file),
                        'filename': checkpoint_file.name,
                        'directory': str(checkpoint_file.parent.name),
                        'dataset': data.get('dataset_name', 'Unknown'),
                        'task': data.get('task', 'Unknown'),
                        'provider': data.get('provider', 'Unknown'),
                        'model': data.get('model', 'Unknown'),
                        'timestamp': data.get('timestamp', 'Unknown'),
                        'progress': data.get('progress', 0),
                        'current_index': data.get('current_index', 0),
                        'total_records': data.get('total_records', 0)
                    }
                    
                    # Apply filters if provided
                    if dataset_name and checkpoint_info['dataset'] != dataset_name:
                        continue
                    if task and checkpoint_info['task'] != task:
                        continue
                    if provider and checkpoint_info['provider'] != provider:
                        continue
                    if model and checkpoint_info['model'] != model:
                        continue
                    
                    # Filter incomplete checkpoints if requested
                    if incomplete_only and checkpoint_info['progress'] >= 1.0:
                        continue
                    
                    checkpoints.append(checkpoint_info)
                    
                except Exception as e:
                    logger.debug(f"Error reading checkpoint {checkpoint_file}: {e}")
                    continue
        
        # Sort by timestamp (most recent first)
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return checkpoints
    
    def ask_checkpoint_preference(self) -> bool:
        """
        Ask user if they want to check for existing checkpoints.
        
        Returns:
            True if user wants to check for checkpoints, False otherwise
        """
        print("\nCheckpoint Options")
        print("-" * 40)
        print("Would you like to check for existing checkpoints?")
        print("This allows you to resume from a previous run.")
        print()
        print("1. Yes, check for checkpoints")
        print("2. No, start fresh")
        
        while True:
            try:
                choice = input("\nSelect option (1-2) [2]: ").strip()
                if choice == "" or choice == "2":
                    return False
                elif choice == "1":
                    return True
                else:
                    print("Please enter 1 or 2")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                raise SystemExit(0)
    
    def choose_checkpoint(self, checkpoints: List[Dict[str, str]]) -> Optional[str]:
        """
        Interactive checkpoint selection.
        
        Args:
            checkpoints: List of available checkpoints
            
        Returns:
            Selected checkpoint path or None for new processing
        """
        print("\nResume from Checkpoint?")
        print("-" * 40)
        
        if not checkpoints:
            print("No existing checkpoints found.")
            return None
        
        print("Found existing checkpoints:")
        for i, cp in enumerate(checkpoints, 1):
            # Format timestamp for display
            try:
                ts = datetime.fromisoformat(cp['timestamp'])
                ts_display = ts.strftime("%Y-%m-%d %H:%M")
            except:
                ts_display = cp['timestamp']
            
            print(f"\n{i}. {cp['filename']}")
            print(f"   Task: {cp['task']}")
            print(f"   Dataset: {cp['dataset']}")
            print(f"   Model: {cp['provider']} - {cp['model']}") 
            print(f"   Progress: {cp['current_index']}/{cp['total_records']} ({cp['progress']:.1%})")
            print(f"   Created: {ts_display}")
        
        print(f"\n0. Start fresh (no checkpoint)")
        
        while True:
            try:
                choice = int(input(f"\nSelect checkpoint (0-{len(checkpoints)}): ").strip())
                if choice == 0:
                    print("Starting fresh processing")
                    return None
                elif 1 <= choice <= len(checkpoints):
                    selected = checkpoints[choice - 1]
                    print(f"\nResuming from: {selected['filename']}")
                    return selected['path']
                else:
                    print(f"Please enter a number between 0 and {len(checkpoints)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                raise SystemExit(0)
    
    def validate_checkpoint_compatibility(self, checkpoint_path: str,
                                        dataset_name: str,
                                        task: str,
                                        provider: str,
                                        model: str) -> bool:
        """
        Validates if checkpoint is compatible with current configuration.
        
        Args:
            checkpoint_path: Path to checkpoint file
            dataset_name: Current dataset name
            task: Current task type  
            provider: Current provider
            model: Current model
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Check compatibility
            mismatches = []
            
            if checkpoint_data.get('dataset_name') != dataset_name:
                mismatches.append(f"Dataset: {checkpoint_data.get('dataset_name')} vs {dataset_name}")
                
            if checkpoint_data.get('task') != task:
                mismatches.append(f"Task: {checkpoint_data.get('task')} vs {task}")
                
            if checkpoint_data.get('provider') != provider:
                mismatches.append(f"Provider: {checkpoint_data.get('provider')} vs {provider}")
                
            if checkpoint_data.get('model') != model:
                mismatches.append(f"Model: {checkpoint_data.get('model')} vs {model}")
            
            if mismatches:
                print("\n⚠️  Checkpoint compatibility issues detected:")
                for mismatch in mismatches:
                    print(f"   - {mismatch}")
                
                response = input("\nProceed anyway? (y/n): ").strip().lower()
                return response == 'y'
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating checkpoint: {e}")
            return False
    
    def load_checkpoint_config(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load configuration from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Configuration dictionary from checkpoint
        """
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Extract relevant configuration
            config = {
                'provider': checkpoint_data.get('provider'),
                'model': checkpoint_data.get('model'),
                'dataset_name': checkpoint_data.get('dataset_name'),
                'dataset_path': checkpoint_data.get('dataset_path'),
                'task': checkpoint_data.get('task'),
                'sample_size': checkpoint_data.get('sample_size'),
                'balanced_sample': checkpoint_data.get('balanced_sample', True),
                'enable_thinking': checkpoint_data.get('enable_thinking', False),
                'use_structure_model': checkpoint_data.get('use_structure_model', False),
                'content_columns': checkpoint_data.get('content_columns'),
                'checkpoint_info': {
                    'current_index': checkpoint_data.get('current_index', 0),
                    'total_records': checkpoint_data.get('total_records', 0),
                    'progress': checkpoint_data.get('progress', 0)
                }
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading checkpoint config: {e}")
            return {}