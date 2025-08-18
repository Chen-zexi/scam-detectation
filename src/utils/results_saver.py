import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from .model_config_manager import ModelConfigManager

class ResultsSaver:
    """Saves evaluation results to a structured, timestamped directory."""
    
    def __init__(self, dataset_name: str, provider: str = None, model: str = None, model_config: Dict[str, Any] = None):
        self.dataset_name = dataset_name
        self.provider = provider
        self.model = model
        self.model_config = model_config or {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"results/evaluation/{dataset_name}/{self.timestamp}")
        
    def save_results(self, 
                    detailed_results: List[Dict[str, Any]], 
                    metrics: Dict[str, Any],
                    dataset_info: Dict[str, Any] = None):
        """
        Save both detailed results and metrics to CSV files
        """
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        detailed_df = pd.DataFrame(detailed_results)
        detailed_path = self.results_dir / "detailed_results.csv"
        detailed_df.to_csv(detailed_path, index=False)
        
        # Save complete evaluation info as JSON for reference
        evaluation_info = {
            'timestamp': self.timestamp,
            'dataset_info': dataset_info,
            'evaluation_config': {
                'provider': self.provider,
                'model': self.model,
                'sample_size': len(detailed_results),
                'model_config': ModelConfigManager.format_for_json(self.model_config) if self.model_config else {}
            },
            'metrics': metrics,
            'results_directory': str(self.results_dir)
        }
        
        
        info_path = self.results_dir / "evaluation_info.json"
        with open(info_path, 'w') as f:
            json.dump(evaluation_info, f, indent=2)
        
        print(f"\nResults saved to: {self.results_dir}")
        print(f"- Detailed results: {detailed_path}")
        print(f"- Evaluation info: {info_path}")
        
        return {
            'results_directory': str(self.results_dir),
            'detailed_results_path': str(detailed_path),
            'info_path': str(info_path)
        }
    
    def create_summary_report(self, metrics: Dict[str, Any], dataset_info: Dict[str, Any] = None) -> str:
        """Create a human-readable summary report"""
        report_lines = [
            "=" * 80,
            "SCAM DETECTION EVALUATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dataset: {self.dataset_name}",
            ""
        ]
        
        if dataset_info:
            scam_count = dataset_info.get('scam_count', 0)
            legitimate_count = dataset_info.get('legitimate_count', 0)
            report_lines.extend([
                "DATASET INFORMATION:",
                f"- Path: {dataset_info.get('path', 'Unknown')}",
                f"- Total records: {dataset_info.get('total_records', 'Unknown')}",
                f"- Content features: {', '.join(dataset_info.get('features', []))}",
                f"- Distribution: {scam_count} scam, {legitimate_count} legitimate",
                ""
            ])
        
        if self.provider and self.model:
            report_lines.extend([
                "MODEL CONFIGURATION:",
                f"- Provider: {self.provider}",
                f"- Model: {self.model}",
            ])
            
            # Add detailed model configuration if available
            if self.model_config:
                model_display = ModelConfigManager.format_for_display(self.model_config)
                # Extract just the parameter lines from the display
                for line in model_display.split('\n'):
                    if 'Reasoning Effort:' in line or 'Verbosity:' in line or 'Temperature:' in line or 'Max Completion Tokens:' in line:
                        report_lines.append(f"  {line.strip()}")
            
            report_lines.append("")
        
        if 'error' not in metrics:
            report_lines.extend([
                "EVALUATION RESULTS:",
                f"- Total samples evaluated: {metrics.get('total_samples_in_dataset', 0)}",
                f"- Successfully processed: {metrics.get('successfully_processed', 0)}",
                f"- Failed predictions: {metrics.get('failed_predictions', 0)}",
                "",
                "PERFORMANCE METRICS:",
                f"- Accuracy: {metrics.get('accuracy', 0):.2%}",
                f"- Precision: {metrics.get('precision', 0):.2%}",
                f"- Recall: {metrics.get('recall', 0):.2%}",
                f"- F1 Score: {metrics.get('f1_score', 0):.2%}",
                f"- Specificity: {metrics.get('specificity', 0):.2%}",
                ""
            ])
            
            cm = metrics.get('confusion_matrix', {})
            report_lines.extend([
                "CONFUSION MATRIX:",
                f"- True Positives (Scam correctly identified): {cm.get('true_positives', 0)}",
                f"- True Negatives (Legitimate correctly identified): {cm.get('true_negatives', 0)}",
                f"- False Positives (Legitimate misclassified as Scam): {cm.get('false_positives', 0)}",
                f"- False Negatives (Scam misclassified as Legitimate): {cm.get('false_negatives', 0)}",
            ])
            
            # Add token usage section if available
            if 'token_usage' in metrics:
                token_usage = metrics['token_usage']
                report_lines.extend([
                    "",
                    "TOKEN USAGE:",
                    f"- Total API Calls: {token_usage.get('total_calls', 0)}",
                    f"- Total Tokens Used: {token_usage.get('total_tokens', 0):,}",
                    f"- Average Tokens per Call: {token_usage.get('average_tokens_per_call', 0):.0f}",
                    f"  - Average Input: {token_usage.get('total_input_tokens', 0) / max(token_usage.get('total_calls', 1), 1):.0f}",
                    f"  - Average Output: {token_usage.get('total_output_tokens', 0) / max(token_usage.get('total_calls', 1), 1):.0f}",
                ])
                
                # Add cached tokens if any
                if token_usage.get('total_cached_tokens', 0) > 0:
                    report_lines.append(f"- Cached Tokens: {token_usage.get('total_cached_tokens', 0):,}")
                
                # Add reasoning tokens if any
                if token_usage.get('total_reasoning_tokens', 0) > 0:
                    report_lines.append(f"- Reasoning Tokens: {token_usage.get('total_reasoning_tokens', 0):,}")
                
                # Add cost estimate if available
                if 'estimated_costs' in token_usage:
                    costs = token_usage['estimated_costs']
                    total_cost = costs.get('total_cost', 0)
                    
                    # Calculate average cost per call
                    total_calls = token_usage.get('total_calls', 1)
                    avg_cost = total_cost / total_calls if total_calls > 0 else 0
                    
                    report_lines.extend([
                        "",
                        "ESTIMATED COST:",
                        f"- Total Cost:     ${total_cost:.4f}",
                        f"- Average/Call:   ${avg_cost:.4f}",
                    ])
                    
                    # Build breakdown with cached cost if present
                    breakdown = f"- Breakdown:      ${costs.get('input_cost', 0):.4f} (input)"
                    if costs.get('cached_cost', 0) > 0:
                        breakdown += f" + ${costs.get('cached_cost', 0):.4f} (cached)"
                    breakdown += f" + ${costs.get('output_cost', 0):.4f} (output)"
                    report_lines.append(breakdown)
                    
                    # Add cache savings if any
                    if costs.get('cache_savings', 0) > 0:
                        report_lines.append(f"- Cache Savings:  ${costs.get('cache_savings', 0):.4f}")
        else:
            report_lines.extend([
                "EVALUATION ERROR:",
                f"- {metrics.get('error', 'Unknown error')}"
            ])
        
        report_lines.append("=" * 80)
        
        # Save report
        report_path = self.results_dir / "summary_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"- Summary report: {report_path}")
        return str(report_path) 