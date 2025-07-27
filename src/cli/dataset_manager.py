"""
Dataset discovery and management functionality.
"""

import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetManager:
    """Handles dataset discovery and selection."""
    
    def __init__(self, data_dir: str = "data/cleaned"):
        """
        Initialize dataset manager.
        
        Args:
            data_dir: Root directory for datasets
        """
        self.data_dir = Path(data_dir)
        
    def discover_datasets(self) -> List[Dict[str, str]]:
        """
        Discovers available datasets in the data directory.
        
        Returns:
            List of dataset information dictionaries
        """
        print("Scanning for available datasets...")
        
        datasets = []
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return datasets
        
        # Look for CSV files in data/ subdirectories
        for csv_file in self.data_dir.rglob("*.csv"):
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
                logger.debug(f"Skipping invalid CSV file {csv_file}: {e}")
                continue
        
        return datasets
    
    def choose_dataset(self, datasets: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Interactive dataset selection.
        
        Args:
            datasets: List of available datasets
            
        Returns:
            Selected dataset information
        """
        print("\nSTEP 2: Select Dataset")
        print("-" * 40)
        
        if not datasets:
            raise ValueError("No datasets available")
        
        print("Available datasets:")
        for i, dataset in enumerate(datasets, 1):
            print(f"\n{i}. {dataset['name']}")
            print(f"   Path: {dataset['path']}")
            print(f"   Records: {dataset['records']:,}")
            print(f"   Columns: {', '.join(dataset['columns'][:5])}", end="")
            if len(dataset['columns']) > 5:
                print(f" ... (+{len(dataset['columns']) - 5} more)")
            else:
                print()
        
        while True:
            try:
                choice = int(input(f"\nSelect dataset (1-{len(datasets)}): ").strip())
                if 1 <= choice <= len(datasets):
                    selected = datasets[choice - 1]
                    print(f"\nSelected: {selected['name']} ({selected['records']:,} records)")
                    return selected
                else:
                    print(f"Please enter a number between 1 and {len(datasets)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                raise SystemExit(0)
    
    def infer_task_from_filename(self, filename: str) -> str:
        """
        Attempts to infer the task type from filename patterns.
        
        Args:
            filename: Name of the dataset file
            
        Returns:
            Inferred task type or 'evaluation' as default
        """
        filename_lower = filename.lower()
        
        # Check for annotation patterns
        if 'annotate' in filename_lower or 'annotation' in filename_lower:
            return 'annotation'
        
        # Check for synthesis patterns  
        if 'synthesis' in filename_lower or 'generated' in filename_lower:
            return 'synthesis'
            
        # Default to evaluation
        return 'evaluation'