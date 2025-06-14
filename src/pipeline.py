#!/usr/bin/env python3
"""
Scam Detection Evaluation Pipeline

A flexible pipeline for evaluating LLMs on scam detection tasks.
Works with any dataset that has a 'label' column (1=scam, 0=legitimate).
Can handle various types of content: emails, texts, conversations, messages, etc.

Usage:
    python pipeline.py --dataset path/to/dataset.csv --provider openai --model gpt-4 --sample-size 100
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to Python path to allow imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from evaluator import ScamDetectionEvaluator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Scam Detection Evaluation Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Path to the dataset CSV file (must contain a "label" column)'
    )
    
    parser.add_argument(
        '--provider', 
        type=str, 
        required=True,
        choices=['openai', 'anthropic', 'gemini', 'local'],
        help='LLM provider to use'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Model name to use (e.g., gpt-4, claude-3-sonnet, gemini-pro, etc.)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=100,
        help='Number of samples to evaluate (default: 100). With --balanced-sample, this is split equally between classes.'
    )
    
    parser.add_argument(
        '--balanced-sample',
        action='store_true',
        help='Sample equal numbers of scam and legitimate messages for balanced evaluation'
    )
    
    parser.add_argument(
        '--random-state', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--content-columns',
        type=str,
        nargs='+',
        help='Specific columns to use as content for evaluation (e.g., --content-columns subject body). If not specified, uses all non-label columns.'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Base directory for output results (default: results)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without making API calls (for testing setup)'
    )
    
    return parser.parse_args()

def validate_dataset(dataset_path: str):
    """Validate that the dataset exists and has required structure"""
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Quick validation by trying to load just the header
    import pandas as pd
    try:
        df_sample = pd.read_csv(dataset_path, nrows=1)
        if 'label' not in df_sample.columns:
            raise ValueError("Dataset must contain a 'label' column")
        print(f"✓ Dataset validation passed")
        print(f"  - File: {dataset_path}")
        print(f"  - Columns: {list(df_sample.columns)}")
    except Exception as e:
        raise ValueError(f"Invalid dataset format: {e}")

def main():
    """Main pipeline execution"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        print("="*80)
        print("SCAM DETECTION EVALUATION PIPELINE")
        print("="*80)
        print(f"Dataset: {args.dataset}")
        print(f"Provider: {args.provider}")
        print(f"Model: {args.model}")
        print(f"Sample size: {args.sample_size}")
        print(f"Balanced sampling: {args.balanced_sample}")
        print(f"Random state: {args.random_state}")
        if args.content_columns:
            print(f"Content columns: {args.content_columns}")
        else:
            print(f"Content columns: All non-label columns")
        
        # Validate dataset
        print(f"\nValidating dataset...")
        validate_dataset(args.dataset)
        
        if args.dry_run:
            print("\n✓ Dry run completed successfully. Dataset and configuration are valid.")
            return
        
        # Initialize evaluator
        evaluator = ScamDetectionEvaluator(
            dataset_path=args.dataset,
            provider=args.provider,
            model=args.model,
            sample_size=args.sample_size,
            balanced_sample=args.balanced_sample,
            random_state=args.random_state,
            content_columns=args.content_columns
        )
        
        # Run full evaluation
        results = evaluator.run_full_evaluation()
        
        # Print final summary
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        
        save_paths = results['save_paths']
        print(f"Results directory: {save_paths['results_directory']}")
        print(f"Detailed results: {save_paths['detailed_results_path']}")
        print(f"Metrics summary: {save_paths['metrics_path']}")
        print(f"Evaluation report: {save_paths['summary_report_path']}")
        
        metrics = results['metrics']
        if 'accuracy' in metrics:
            print(f"\nQuick Stats:")
            print(f"- Accuracy: {metrics['accuracy']:.2%}")
            print(f"- Precision: {metrics['precision']:.2%}")
            print(f"- Recall: {metrics['recall']:.2%}")
            print(f"- F1 Score: {metrics['f1_score']:.2%}")
        
        print(f"\n✓ Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 