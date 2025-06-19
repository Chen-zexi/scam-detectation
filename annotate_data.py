#!/usr/bin/env python3
"""
Example usage of the LLM Annotation Pipeline

This script demonstrates how to use the annotation pipeline to generate explanations
for why content is classified as scam or legitimate.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.annotation_pipeline import LLMAnnotationPipeline

def annotation():
    print("="*80)
    print("ANNOTATION: Generating explanations for scam/legitimate classifications")
    print("="*80)
    
    dataset_path = "data/cleaned/unified_error_dataset/unified_error_dataset.csv"
    provider = "lm-studio"
    model = "unsloth/qwen3-235b-a22b"
    sample_size = 2540
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please ensure the dataset file exists in the current directory.")
        return
    
    try:
        # Initialize annotation pipeline
        pipeline = LLMAnnotationPipeline(
            dataset_path=dataset_path,
            provider=provider,
            model=model,
            sample_size=sample_size,
            random_state=42,
            content_columns=['content'],
            balanced_sample=False,
            output_dir="results/annotated"
        )
        
        # Run annotation
        results = pipeline.run_full_annotation()
        
        print("\n Annotation completed successfully!")
        print(f"Results saved to: {results['save_paths']['results_directory']}")
        
    except Exception as e:
        print(f" Annotation failed: {e}")

def main():
    """Run annotation example"""
    print("SCAM DETECTION ANNOTATION PIPELINE - DATASET")
    annotation()

if __name__ == "__main__":
    main() 