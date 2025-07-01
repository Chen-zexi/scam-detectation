#!/usr/bin/env python3
"""
Dataset Inspection Tool for LLM Configuration

This script analyzes datasets to estimate token usage and provide recommendations
for LLM context length configuration and processing requirements.

Features:
- Analyze token distribution across datasets
- Generate context length recommendations for different models
- Filter datasets by maximum token limit and create capped versions
- Estimate processing costs and requirements

Usage:
    1. Manual Configuration (edit MANUAL_CONFIG below):
       python dataset_inspection.py
    
    2. Command Line Arguments (overrides manual config):
       python dataset_inspection.py --dataset path/to/dataset.csv
       python dataset_inspection.py --dataset path/to/dataset.csv --content-columns subject body
       python dataset_inspection.py --dataset path/to/dataset.csv --task annotation --provider openai
       
    3. Create a filtered dataset with token limit:
       python dataset_inspection.py --dataset path/to/dataset.csv --max-tokens 2000
       # This will create a new file: path/to/dataset_cap.csv with records <= 2000 tokens
"""

# =============================================================================
# MANUAL CONFIGURATION SECTION
# =============================================================================
# Set USE_MANUAL_CONFIG = True to use the configuration below
# Set USE_MANUAL_CONFIG = False to use command line arguments only

USE_MANUAL_CONFIG = True  # Change to True to enable manual configuration

MANUAL_CONFIG = {
    'dataset': 'data/cleaned/unified_phishing_email_dataset.csv',  # Path to your dataset
    'content_columns': ['subject', 'body'],  # e.g., ['subject', 'body'] or None for auto-detection
    'task': 'annotation',  # 'evaluation' or 'annotation'
    'provider': 'openai',  # 'openai', 'anthropic', or 'gemini'
    'output': None,  # Custom output file path or None for auto-generated
    'save_report': False,  # Whether to save detailed results to file
    'max_tokens': 8000,  # Maximum tokens per record (None = no filtering, int = create capped dataset)
}

# =============================================================================

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Using approximate token estimation.")

class DatasetInspector:
    """Analyze datasets for token usage and LLM configuration recommendations"""
    
    def __init__(self, dataset_path: str, content_columns: Optional[List[str]] = None, max_tokens: Optional[int] = None):
        self.dataset_path = dataset_path
        self.content_columns = content_columns
        self.max_tokens = max_tokens
        self.df = None
        self.original_df = None  # Store original before filtering
        self.tokenizer = None
        self.analysis_results = {}
        self.filtered_records = 0  # Track how many records were filtered out
        
        # Load tokenizer if available
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
                self.tokenizer = None
    
    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset and perform basic validation"""
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"✓ Dataset loaded successfully: {len(self.df):,} records")
            print(f"  Columns: {list(self.df.columns)}")
            
            # Identify label column
            if 'label' in self.df.columns:
                print(f"  Label distribution: {dict(self.df['label'].value_counts())}")
            
            return self.df
            
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
    
    def identify_content_columns(self) -> List[str]:
        """Automatically identify content columns if not specified"""
        if self.content_columns:
            # Validate specified columns exist
            missing_cols = [col for col in self.content_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Specified content columns not found: {missing_cols}")
            return self.content_columns
        
        # Auto-detect content columns (exclude likely metadata columns)
        exclude_columns = {'id', 'label', 'index', 'timestamp', 'date', 'source', 'dataset'}
        text_columns = []
        
        for col in self.df.columns:
            if col.lower() not in exclude_columns and self.df[col].dtype == 'object':
                # Check if column contains substantial text
                sample_text = str(self.df[col].iloc[0]) if not pd.isna(self.df[col].iloc[0]) else ""
                if len(sample_text) > 10:  # Assume columns with >10 chars are content
                    text_columns.append(col)
        
        if not text_columns:
            # Fallback: use all non-excluded object columns
            text_columns = [col for col in self.df.columns 
                          if col.lower() not in exclude_columns and self.df[col].dtype == 'object']
        
        print(f"  Auto-detected content columns: {text_columns}")
        return text_columns
    
    def filter_dataset_by_tokens(self, content_columns: List[str]) -> pd.DataFrame:
        """Filter dataset to remove records exceeding max token limit"""
        if not self.max_tokens:
            return self.df
        
        print(f"\n" + "="*60)
        print(f"FILTERING DATASET BY TOKEN LIMIT: {self.max_tokens:,}")
        print("="*60)
        
        # Store original dataset
        self.original_df = self.df.copy()
        original_count = len(self.original_df)
        
        # Calculate tokens for each record and filter
        filtered_indices = []
        filtered_out_count = 0
        
        print("Analyzing records for token filtering...")
        for idx, row in self.df.iterrows():
            if idx % 1000 == 0 and idx > 0:
                print(f"  Processed {idx:,}/{len(self.df):,} records")
            
            record_tokens = 0
            for col in content_columns:
                col_tokens = self.estimate_tokens(row[col])
                record_tokens += col_tokens
            
            if record_tokens <= self.max_tokens:
                filtered_indices.append(idx)
            else:
                filtered_out_count += 1
        
        # Create filtered dataset
        self.df = self.df.loc[filtered_indices].reset_index(drop=True)
        self.filtered_records = filtered_out_count
        
        # Save filtered dataset
        capped_dataset_path = self._save_capped_dataset()
        
        # Print filtering results
        filtered_count = len(self.df)
        retention_rate = (filtered_count / original_count) * 100
        
        print(f"\nFiltering Results:")
        print(f"  Original records: {original_count:,}")
        print(f"  Records after filtering: {filtered_count:,}")
        print(f"  Records removed: {filtered_out_count:,}")
        print(f"  Retention rate: {retention_rate:.1f}%")
        print(f"  Capped dataset saved: {capped_dataset_path}")
        
        return self.df
    
    def _save_capped_dataset(self) -> str:
        """Save the filtered dataset with _cap suffix"""
        original_path = Path(self.dataset_path)
        capped_filename = f"{original_path.stem}_cap{original_path.suffix}"
        capped_path = original_path.parent / capped_filename
        
        # Save the filtered dataset
        self.df.to_csv(capped_path, index=False)
        
        return str(capped_path)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text"""
        if not text or pd.isna(text):
            return 0
            
        text = str(text)
        
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback: rough estimation (1 token ≈ 4 characters for English)
        return max(1, len(text) // 4)
    
    def analyze_token_distribution(self, content_columns: List[str]) -> Dict[str, Any]:
        """Analyze token distribution across the dataset"""
        print("\n" + "="*60)
        print("TOKEN ANALYSIS")
        print("="*60)
        
        # Calculate tokens for each record
        token_data = []
        
        print("Analyzing token usage...")
        for idx, row in self.df.iterrows():
            if idx % 1000 == 0 and idx > 0:
                print(f"  Processed {idx:,}/{len(self.df):,} records")
            
            record_tokens = 0
            for col in content_columns:
                col_tokens = self.estimate_tokens(row[col])
                record_tokens += col_tokens
            
            token_data.append(record_tokens)
        
        token_series = pd.Series(token_data)
        
        # Calculate statistics
        stats = {
            'total_records': len(self.df),
            'content_columns': content_columns,
            'token_stats': {
                'mean': float(token_series.mean()),
                'median': float(token_series.median()),
                'std': float(token_series.std()),
                'min': int(token_series.min()),
                'max': int(token_series.max()),
                'p95': float(token_series.quantile(0.95)),
                'p99': float(token_series.quantile(0.99))
            },
            'total_input_tokens': int(token_series.sum())
        }
        
        # Print detailed statistics
        print(f"\nInput Token Statistics:")
        print(f"  Total records: {stats['total_records']:,}")
        print(f"  Content columns: {', '.join(content_columns)}")
        print(f"  Total input tokens: {stats['total_input_tokens']:,}")
        print(f"  Average tokens per record: {stats['token_stats']['mean']:.1f}")
        print(f"  Median tokens per record: {stats['token_stats']['median']:.1f}")
        print(f"  Standard deviation: {stats['token_stats']['std']:.1f}")
        print(f"  Min tokens: {stats['token_stats']['min']:,}")
        print(f"  Max tokens: {stats['token_stats']['max']:,}")
        print(f"  95th percentile: {stats['token_stats']['p95']:.1f}")
        print(f"  99th percentile: {stats['token_stats']['p99']:.1f}")
        
        # Distribution analysis
        ranges = [
            (0, 100, "Very Short"),
            (100, 500, "Short"),
            (500, 1000, "Medium"),
            (1000, 2000, "Long"),
            (2000, 4000, "Very Long"),
            (4000, 8000, "X Long"),
            (8000, 10000, "XX Long"),
            (10000, float('inf'), "XXX Long")
        ]
        
        print(f"\nToken Length Distribution:")
        for min_tokens, max_tokens, label in ranges:
            if max_tokens == float('inf'):
                count = len(token_series[token_series >= min_tokens])
                percentage = (count / len(token_series)) * 100
                print(f"  {label} (≥{min_tokens:,}): {count:,} ({percentage:.1f}%)")
            else:
                count = len(token_series[(token_series >= min_tokens) & (token_series < max_tokens)])
                percentage = (count / len(token_series)) * 100
                print(f"  {label} ({min_tokens}-{max_tokens-1}): {count:,} ({percentage:.1f}%)")
        
        return stats
    
    def estimate_processing_requirements(self, stats: Dict[str, Any], 
                                       task_type: str = "evaluation", 
                                       provider: str = "openai") -> Dict[str, Any]:
        """Estimate processing requirements and costs"""
        print("\n" + "="*60)
        print("PROCESSING REQUIREMENTS")
        print("="*60)
        
        total_input_tokens = stats['total_input_tokens']
        
        # Estimate output tokens based on task type
        if task_type == "annotation":
            # Annotation tasks generate detailed explanations
            avg_output_per_record = 200  # tokens
        else:  # evaluation
            # Evaluation tasks generate shorter responses
            avg_output_per_record = 50   # tokens
        
        total_output_tokens = stats['total_records'] * avg_output_per_record
        total_tokens = total_input_tokens + total_output_tokens
        
        # Provider-specific estimates
        cost_estimates = self._get_cost_estimates(provider, total_input_tokens, total_output_tokens)
        
        requirements = {
            'task_type': task_type,
            'provider': provider,
            'input_tokens': total_input_tokens,
            'estimated_output_tokens': total_output_tokens,
            'total_tokens': total_tokens,
            'cost_estimates': cost_estimates
        }
        
        print(f"Task Type: {task_type.title()}")
        print(f"Provider: {provider}")
        print(f"Total input tokens: {total_input_tokens:,}")
        print(f"Estimated output tokens: {total_output_tokens:,}")
        print(f"Total tokens: {total_tokens:,}")
        
        if cost_estimates:
            print(f"\nCost Estimates ({provider}):")
            for model, cost in cost_estimates.items():
                print(f"  {model}: ${cost:.2f}")
        
        return requirements
    
    def _get_cost_estimates(self, provider: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Get cost estimates for different models"""
        # Pricing per 1M tokens
        pricing = {
            'openai': {
                'gpt-4.1': {'input': 2, 'output': 8},
                'gpt-4.1-mini': {'input': 0.4, 'output': 1.6}
            }
        }
        
        if provider not in pricing:
            return {}
        
        estimates = {}
        for model, prices in pricing[provider].items():
            input_cost = (input_tokens / 1_000_000) * prices['input']
            output_cost = (output_tokens / 1_000_000) * prices['output']
            total_cost = input_cost + output_cost
            estimates[model] = total_cost
        
        return estimates
    
    def get_context_recommendations(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Provide context length recommendations"""
        print("\n" + "="*60)
        print("CONTEXT LENGTH RECOMMENDATIONS")
        print("="*60)
        
        max_tokens = stats['token_stats']['max']
        p95_tokens = stats['token_stats']['p95']
        p99_tokens = stats['token_stats']['p99']
        
        # Add system prompt overhead (estimated)
        system_prompt_tokens = 200  # Estimated system prompt tokens
        output_buffer = 500  # Buffer for output tokens
        
        # Calculate recommended context lengths
        recommended_min = int(max_tokens + system_prompt_tokens + output_buffer)
        recommended_safe = int(p99_tokens + system_prompt_tokens + output_buffer)
        recommended_optimal = int(p95_tokens + system_prompt_tokens + output_buffer)
        
        # Model context limits
        model_limits = {
            'gpt-4.1': 128000,
            'qwen3-30b-a3b': 38000,
        }
        
        recommendations = {
            'input_stats': {
                'max_input_tokens': max_tokens,
                'p95_input_tokens': p95_tokens,
                'p99_input_tokens': p99_tokens
            },
            'overhead': {
                'system_prompt_tokens': system_prompt_tokens,
                'output_buffer': output_buffer
            },
            'recommended_context_lengths': {
                'minimum': recommended_min,
                'safe_p99': recommended_safe,
                'optimal_p95': recommended_optimal
            },
            'model_compatibility': {}
        }
        
        print(f"Input Token Analysis:")
        print(f"  Maximum input tokens: {max_tokens:,}")
        print(f"  95th percentile: {p95_tokens:.0f}")
        print(f"  99th percentile: {p99_tokens:.0f}")
        
        print(f"\nOverhead Estimates:")
        print(f"  System prompt: ~{system_prompt_tokens} tokens")
        print(f"  Output buffer: ~{output_buffer} tokens")
        
        print(f"\nRecommended Context Lengths:")
        print(f"  Minimum (handles all records): {recommended_min:,}")
        print(f"  Safe (handles 99% of records): {recommended_safe:,}")
        print(f"  Optimal (handles 95% of records): {recommended_optimal:,}")
        
        print(f"\nModel Compatibility:")
        for model, limit in model_limits.items():
            if recommended_min <= limit:
                status = "Compatible"
            elif recommended_safe <= limit:
                status = "Mostly Compatible (99%)"
            elif recommended_optimal <= limit:
                status = "Partially Compatible (95%)"
            else:
                status = "Too Small"
            
            recommendations['model_compatibility'][model] = {
                'limit': limit,
                'status': status,
                'compatible': recommended_min <= limit
            }
            print(f"  {model:20} ({limit:8,}): {status}")
        
        return recommendations
    
    def generate_summary_report(self, stats: Dict[str, Any], 
                              requirements: Dict[str, Any], 
                              recommendations: Dict[str, Any],
                              filtering_info: Optional[Dict[str, Any]] = None) -> str:
        """Generate a comprehensive summary report"""
        report = f"""
DATASET INSPECTION SUMMARY REPORT
{'='*50}

Dataset Information:
  File: {self.dataset_path}
  Total Records: {stats['total_records']:,}
  Content Columns: {', '.join(stats['content_columns'])}"""
        
        if filtering_info:
            report += f"""
  
Token Filtering Applied:
  Max Tokens Limit: {filtering_info['max_tokens']:,}
  Original Records: {filtering_info['original_records']:,}
  Records After Filtering: {filtering_info['filtered_records']:,}
  Records Removed: {filtering_info['removed_records']:,}
  Retention Rate: {filtering_info['retention_rate']:.1f}%
  Capped Dataset: {Path(filtering_info['capped_dataset_path']).name}"""
        
        report += f"""

Token Analysis:
  Total Input Tokens: {stats['total_input_tokens']:,}
  Average per Record: {stats['token_stats']['mean']:.1f}
  Maximum Record: {stats['token_stats']['max']:,} tokens
  95th Percentile: {stats['token_stats']['p95']:.0f} tokens

Processing Requirements:
  Task: {requirements['task_type'].title()}
  Estimated Output Tokens: {requirements['estimated_output_tokens']:,}
  Total Processing Tokens: {requirements['total_tokens']:,}

Context Length Recommendations:
  Minimum Required: {recommendations['recommended_context_lengths']['minimum']:,}
  Safe (99%): {recommendations['recommended_context_lengths']['safe_p99']:,}
  Optimal (95%): {recommendations['recommended_context_lengths']['optimal_p95']:,}

Compatible Models:
"""
        
        for model, info in recommendations['model_compatibility'].items():
            status = "✓" if info['compatible'] else "❌"
            report += f"  {status} {model} ({info['limit']:,} context)\n"
        
        if requirements['cost_estimates']:
            report += f"\nEstimated Costs ({requirements['provider']}):\n"
            for model, cost in requirements['cost_estimates'].items():
                report += f"  {model}: ${cost:.2f}\n"
        
        return report
    
    def save_results(self, output_path: str = None):
        """Save analysis results to JSON file"""
        if not output_path:
            dataset_name = Path(self.dataset_path).stem
            output_path = f"{dataset_name}_inspection_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {output_path}")
    
    def run_full_inspection(self, task_type: str = "evaluation", 
                           provider: str = "openai", 
                           save_report: bool = True) -> Dict[str, Any]:
        """Run complete dataset inspection"""
        print("="*70)
        print("DATASET INSPECTION FOR LLM CONFIGURATION")
        print("="*70)
        print(f"Dataset: {self.dataset_path}")
        if self.max_tokens:
            print(f"Token limit: {self.max_tokens:,}")
        
        # Load dataset
        self.load_dataset()
        
        # Identify content columns
        content_columns = self.identify_content_columns()
        
        # Filter dataset if max_tokens is specified
        if self.max_tokens:
            self.filter_dataset_by_tokens(content_columns)
        
        # Analyze tokens
        stats = self.analyze_token_distribution(content_columns)
        
        # Estimate processing requirements
        requirements = self.estimate_processing_requirements(stats, task_type, provider)
        
        # Get context recommendations
        recommendations = self.get_context_recommendations(stats)
        
        # Store results
        filtering_info = None
        if self.max_tokens and self.original_df is not None:
            # Get the capped dataset path
            original_path = Path(self.dataset_path)
            capped_filename = f"{original_path.stem}_cap{original_path.suffix}"
            capped_path = original_path.parent / capped_filename
            
            filtering_info = {
                'max_tokens': self.max_tokens,
                'original_records': len(self.original_df),
                'filtered_records': len(self.df),
                'removed_records': self.filtered_records,
                'retention_rate': (len(self.df) / len(self.original_df)) * 100,
                'capped_dataset_path': str(capped_path)
            }
        
        self.analysis_results = {
            'dataset_info': {
                'path': self.dataset_path,
                'records': len(self.df),
                'columns': list(self.df.columns),
                'content_columns': content_columns,
                'filtering_applied': filtering_info
            },
            'token_analysis': stats,
            'processing_requirements': requirements,
            'context_recommendations': recommendations,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Generate and print summary
        summary = self.generate_summary_report(stats, requirements, recommendations, filtering_info)
        print(summary)
        
        # Save results
        if save_report:
            self.save_results()
        
        return self.analysis_results

def get_configuration():
    """Get configuration from manual config or command line arguments"""
    if USE_MANUAL_CONFIG:
        print("Using manual configuration from MANUAL_CONFIG")
        return type('Args', (), {
            'dataset': MANUAL_CONFIG['dataset'],
            'content_columns': MANUAL_CONFIG['content_columns'],
            'task': MANUAL_CONFIG['task'],
            'provider': MANUAL_CONFIG['provider'],
            'output': MANUAL_CONFIG['output'],
            'no_save': not MANUAL_CONFIG['save_report'],
            'max_tokens': MANUAL_CONFIG['max_tokens']
        })()
    else:
        return parse_arguments()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Inspect dataset for LLM token usage and context length requirements",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=not USE_MANUAL_CONFIG,
        help='Path to the dataset CSV file'
    )
    
    parser.add_argument(
        '--content-columns',
        type=str,
        nargs='+',
        help='Specific columns to analyze for content (e.g., --content-columns subject body)'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['evaluation', 'annotation'],
        default='evaluation',
        help='Type of task to estimate output tokens for (default: evaluation)'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'anthropic', 'gemini'],
        default='openai',
        help='LLM provider for cost estimation (default: openai)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for detailed results (default: auto-generated)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving detailed results to file'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        help='Maximum tokens per record (records exceeding this will be filtered out and a new _cap dataset created)'
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    try:
        args = get_configuration()
        
        # Validate dataset exists
        if not Path(args.dataset).exists():
            print(f"Dataset not found: {args.dataset}")
            sys.exit(1)
        
        # Create inspector
        inspector = DatasetInspector(
            dataset_path=args.dataset,
            content_columns=args.content_columns,
            max_tokens=args.max_tokens
        )
        
        # Run inspection
        results = inspector.run_full_inspection(
            task_type=args.task,
            provider=args.provider,
            save_report=not args.no_save
        )
        
        if args.output and not args.no_save:
            inspector.save_results(args.output)
        
        print("\nDataset inspection completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nInspection interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during inspection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 