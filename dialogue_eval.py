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

def dialogue_eval():
    print("="*80)
    print("DIALOGUE EVALUATION")
    print("="*80)
    
    dataset_path = "data/cleaned/phising_dialogue_dataset.csv"
    provider = "lm-studio"
    model = "unsloth/qwen3-30b-a3b"
    sample_size = 1600
    
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
            content_columns=['dialogue'],
            balanced_sample=False
        )
        
        # Run evaluation
        results = evaluator.run_full_evaluation()
        
        print("\n✓ Dialogue Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Dialogue Evaluation failed: {e}")

def main():
    """Run all examples"""
    print("SCAM DETECTION PIPELINE - DIALOGUE EVALUATION")
    dialogue_eval()
    
    

if __name__ == "__main__":
    main() 