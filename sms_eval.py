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

def email_eval():
    """Example using the unified dataset with all features"""
    print("="*80)
    print("EMAIL EVALUATION: Unified Dataset - All Content Features (Local Model)")
    print("="*80)
    
    dataset_path = "phishing_sms_dataset.csv"
    provider = "lm-studio"
    model = "qwen3-235b-a22b-128k"
    sample_size = 100
    
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
            balanced_sample=True
        )
        
        # Run evaluation
        results = evaluator.run_full_evaluation()
        
        print("\n✓ Email Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Email Evaluation failed: {e}")

def main():
    """Run all examples"""
    print("SCAM DETECTION PIPELINE - EMAIL EVALUATION")
    email_eval()
    
    

if __name__ == "__main__":
    main() 