#!/usr/bin/env python3
"""
Example usage of the Scam Detection Evaluation Pipeline

This script demonstrates how to use the pipeline with different types of content for scam detection.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from src import ScamDetectionEvaluator

def error_eval():
    print("="*80)
    print("ERROR EVALUATION: Unified Dataset - All Content Features (Local Model)")
    print("="*80)
    
    dataset_path = "data/cleaned/unified_error_dataset/unified_error_dataset.csv"
    provider = "openai"
    model = "gpt-4.1-mini"
    sample_size = 2540
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please ensure the dataset file exists in the data/cleaned")
        return
    
    try:
        # Initialize evaluator - uses all content features by default
        evaluator = ScamDetectionEvaluator(
            dataset_path=dataset_path,
            provider=provider,
            model=model,
            sample_size=sample_size,
            random_state=42,
            content_columns=['content'],
            balanced_sample=False
        )
        
        # Run evaluation
        results = evaluator.run_full_evaluation()
        
        print("\nError Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error Evaluation failed: {e}")

def main():
    """Run all examples"""
    print("SCAM DETECTION PIPELINE - ERROR EVALUATION")
    error_eval()
    
    

if __name__ == "__main__":
    main() 