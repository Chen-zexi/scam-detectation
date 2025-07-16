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

from src.evaluate import ScamDetectionEvaluator

def email_eval():
    print("="*80)
    print("EMAIL EVALUATION")
    print("="*80)
    
    dataset_path = "data/cleaned/unified_phishing_email_dataset.csv"
    provider = "openai"
    model = "gpt-4.1-mini"
    sample_size = 10
    
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
            content_columns=['subject', 'body'],
            use_structure_model=True,
            enable_thinking=True
        )
        
        # Run evaluation
        results = evaluator.run_full_evaluation()
        
        print("\nEmail Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Email Evaluation failed: {e}")

def main():
    """Run all examples"""
    print("SCAM DETECTION PIPELINE - EMAIL EVALUATION")
    email_eval()
    
    

if __name__ == "__main__":
    main() 