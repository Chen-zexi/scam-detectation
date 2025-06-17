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

def sms_eval():
    print("="*80)
    print("SMS EVALUATION")
    print("="*80)
    
    dataset_path = "phishing_sms_dataset.csv"
    provider = "lm-studio"
    model = "unsloth/qwen3-30b-a3b"
    sample_size = 5572
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please ensure the dataset file exists in the current directory.")
        return
    
    try:
        # Initialize evaluator - uses all content features by default
        evaluator = ScamDetectionEvaluator(
            dataset_path=dataset_path,
            provider=provider,
            model=model,
            sample_size=sample_size,
            random_state=42,
            content_columns=['message'],
            balanced_sample=False
        )
        
        # Run evaluation
        results = evaluator.run_full_evaluation()
        
        print("\n✓ SMS Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ SMS Evaluation failed: {e}")

def main():
    """Run all examples"""
    print("SCAM DETECTION PIPELINE - SMS EVALUATION")
    sms_eval()
    
    

if __name__ == "__main__":
    main() 