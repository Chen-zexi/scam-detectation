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

def example():
    """Example using the unified dataset with all features"""
    print("="*80)
    print("EXAMPLE 1: Unified Dataset - All Content Features (Local Model)")
    print("="*80)
    
    dataset_path = "unified_phishing_email_dataset.csv"
    provider = "gemini"
    model = "gemini-2.5-flash-preview-05-20"
    sample_size = 5
    
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
            content_columns=['subject', 'body']
        )
        
        # Run evaluation
        results = evaluator.run_full_evaluation()
        
        print("\n✓ Example 1 completed successfully!")
        
    except Exception as e:
        print(f"❌ Example 1 failed: {e}")

def example_with_specific_content_columns():
    """Example specifying which columns to use as content"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Unified Dataset - Specific Content Features")
    print("="*80)
    
    dataset_path = "unified_phishing_email_dataset.csv"
    provider = "openai"
    model = "gpt-4.1"
    sample_size = 5
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        return
    
    try:
        # Only use body as content, ignore other subject.
        evaluator = ScamDetectionEvaluator(
            dataset_path=dataset_path,
            provider=provider,
            model=model,
            sample_size=sample_size,
            random_state=42,
            content_columns=['body']  # Specify which columns to use
        )
        
        results = evaluator.run_full_evaluation()
        print("\n✓ Example 2 completed successfully!")
        
    except Exception as e:
        print(f"❌ Example 2 failed: {e}")

def show_usage_options():
    """Show different ways to use the pipeline"""
    print("\n" + "="*80)
    print("USAGE OPTIONS")
    print("="*80)
    
    print("1. Command Line Usage (All Content Features):")
    print("   python src/pipeline.py --dataset data.csv --provider openai --model gpt-4.1")
    
    print("\n2. Command Line Usage (Specific Content Features):")
    print("   python src/pipeline.py --dataset data.csv --provider openai --model gpt-4.1 --content-columns message sender")
    
    print("\n3. Python Script Usage (All Features):")
    print("""
   from src import ScamDetectionEvaluator
   
   evaluator = ScamDetectionEvaluator(
       dataset_path='data.csv',
       provider='openai',
       model='gpt-4.1',
       sample_size=100
   )
   
   results = evaluator.run_full_evaluation()
   """)
    
    print("\n4. Python Script Usage (Specific Content Columns):")
    print("""
   evaluator = ScamDetectionEvaluator(
       dataset_path='data.csv',
       provider='openai',
       model='gpt-4.1',
       sample_size=100,
       content_columns=['message', 'sender']  # Only use these
   )
   
   results = evaluator.run_full_evaluation()
   """)
    

def main():
    """Run all examples"""
    print("SCAM DETECTION PIPELINE - EXAMPLES")
    
    show_usage_options()
    
    # Run examples
    example()
    example_with_specific_content_columns()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main() 