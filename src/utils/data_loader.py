import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any

class DatasetLoader:
    """
    Generic dataset loader that can handle different datasets with varying features.
    Requires only a 'label' column with 1=scam, 0=legitimate.
    Designed for scam detection across various content types (emails, texts, conversations, etc.).
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset_name = Path(dataset_path).stem
        self.df = None
        self.features = []
        
    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded with {len(self.df)} records")
            print(f"Columns: {list(self.df.columns)}")
            
            # Validate required column
            if 'label' not in self.df.columns:
                raise ValueError("Dataset must contain a 'label' column")
                
            # Fill NaN values with empty strings
            self.df = self.df.fillna("")
            
            # Clean and convert label column with error handling
            original_count = len(self.df)
            
            # Convert labels to numeric, replacing invalid values with NaN
            self.df['label'] = pd.to_numeric(self.df['label'], errors='coerce')
            
            # Remove rows with invalid labels (NaN)
            invalid_labels = self.df['label'].isna()
            if invalid_labels.any():
                invalid_count = invalid_labels.sum()
                print(f"Warning: Found {invalid_count} rows with invalid labels, removing them")
                self.df = self.df.dropna(subset=['label']).reset_index(drop=True)
            
            # Convert to integer
            self.df['label'] = self.df['label'].astype(int)
            
            # Validate label values are 0 or 1
            valid_labels = self.df['label'].isin([0, 1])
            if not valid_labels.all():
                invalid_values = self.df[~valid_labels]['label'].unique()
                print(f"Warning: Found invalid label values {invalid_values}, removing rows")
                self.df = self.df[valid_labels].reset_index(drop=True)
            
            final_count = len(self.df)
            if final_count != original_count:
                print(f"Dataset cleaned: {original_count} -> {final_count} records ({original_count - final_count} removed)")
            
            print(f"Label column processed successfully")
            print(f"Valid label values: {sorted(self.df['label'].unique())}")
            
            # Identify features (exclude label and common non-content columns)
            excluded_columns = {'label', 'id', 'source', 'index', 'idx', 'row_id', 'record_id'}
            self.features = [col for col in self.df.columns if col.lower() not in excluded_columns]
            
            print(f"Identified content features: {self.features}")
            
            # More descriptive label distribution
            label_dist = self.df['label'].value_counts().to_dict()
            scam_count = label_dist.get(1, 0)
            legitimate_count = label_dist.get(0, 0)
            print(f"Label distribution: {scam_count} scam, {legitimate_count} legitimate")
            
            return self.df
            
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")
    
    def sample_data(self, n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
        """Sample n records from the dataset for evaluation"""
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        if len(self.df) < n_samples:
            print(f"Dataset has only {len(self.df)} records, using all records")
            return self.df.reset_index(drop=True)
            
        sample_df = self.df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
        
        # Show sample distribution
        sample_dist = sample_df['label'].value_counts().to_dict()
        scam_count = sample_dist.get(1, 0)
        legitimate_count = sample_dist.get(0, 0)
        print(f"Sampled {len(sample_df)} records: {scam_count} scam, {legitimate_count} legitimate")
        
        return sample_df
    
    def sample_balanced_data(self, n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
        """
        Sample balanced data with equal numbers of scam and legitimate messages
        
        Args:
            n_samples: Total number of samples (will be split equally between classes)
            random_state: Random seed for reproducibility
            
        Returns:
            Balanced sample dataframe
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Split data by class
        scam_df = self.df[self.df['label'] == 1]
        legitimate_df = self.df[self.df['label'] == 0]
        
        print(f"Available data: {len(scam_df)} scam, {len(legitimate_df)} legitimate")
        
        # Calculate samples per class (half of total sample size)
        samples_per_class = n_samples // 2
        
        # Check if we have enough samples of each class
        max_scam_samples = min(samples_per_class, len(scam_df))
        max_legitimate_samples = min(samples_per_class, len(legitimate_df))
        
        # Use the minimum to ensure balance
        final_samples_per_class = min(max_scam_samples, max_legitimate_samples)
        
        if final_samples_per_class < samples_per_class:
            print(f"Warning: Requested {samples_per_class} samples per class, but can only provide {final_samples_per_class}")
            print(f"Limited by minority class with {min(len(scam_df), len(legitimate_df))} samples")
        
        # Sample equal numbers from each class
        np.random.seed(random_state)
        scam_sample = scam_df.sample(n=final_samples_per_class, random_state=random_state)
        legitimate_sample = legitimate_df.sample(n=final_samples_per_class, random_state=random_state + 1)
        
        # Combine and shuffle
        balanced_sample = pd.concat([scam_sample, legitimate_sample], ignore_index=True)
        balanced_sample = balanced_sample.sample(frac=1, random_state=random_state + 2).reset_index(drop=True)
        
        print(f"Balanced sample created: {final_samples_per_class} scam, {final_samples_per_class} legitimate")
        print(f"Total balanced sample size: {len(balanced_sample)}")
        
        return balanced_sample
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        label_dist = self.df['label'].value_counts().to_dict()
        return {
            'name': self.dataset_name,
            'path': self.dataset_path,
            'total_records': len(self.df),
            'features': self.features,
            'label_distribution': label_dist,
            'scam_count': label_dist.get(1, 0),
            'legitimate_count': label_dist.get(0, 0)
        } 