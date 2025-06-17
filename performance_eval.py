#!/usr/bin/env python3
"""
Example usage of the Performance Evaluation Pipeline

This script demonstrates different ways to use the performance evaluation pipeline
to measure token/second performance across different models and providers.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.performance_evaluator import PerformanceEvaluator

def performance_evaluation():
    """
    Performance evaluation
    """
    print("\n" + "="*80)
    print("PERFORMANCE EVALUATION")
    print("="*80)
    
    # Check if we have a dataset available
    dataset_paths = [
        "unified_error_dataset/unified_error_dataset.csv"
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if Path(path).exists():
            dataset_path = path
            break
    
    if not dataset_path:
        print("No dataset found.")
        return
    
    model_configs = [
        {"provider": "openai", "model": "gpt-4.1-mini"},
        {"provider": "openai", "model": "gpt-4.1"},
        {"provider": "lm-studio", "model": "unsloth/qwen3-30b-a3b"},
        # Add more models as needed
    ]
    
    try:
        # Initialize evaluator with dataset
        evaluator = PerformanceEvaluator(
            dataset_path=dataset_path,
            sample_size=5,
            random_state=42,
            content_columns=['content']
        )
        
        # Run evaluation
        results = evaluator.evaluate_multiple_models(model_configs)
        
        # Print results
        evaluator.print_performance_report(results)
        
        # Save results
        save_paths = evaluator.save_results(results, "performance_results")
        print(f"\n✓ Results saved to: {save_paths['output_directory']}")
        
    except Exception as e:
        print(f"❌ Performance evaluation failed: {e}")


def main():
    """Run performance evaluation"""
    print("PERFORMANCE EVALUATION PIPELINE")
    print()
    
    performance_evaluation()
    
    print("\n" + "="*80)
    print("PERFORMANCE EVALUATION COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main() 