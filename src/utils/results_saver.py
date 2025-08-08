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
        
        # Prepare metrics for CSV format
        metrics_data = self._flatten_metrics_for_csv(metrics, dataset_info)
        metrics_df = pd.DataFrame([metrics_data])
        metrics_path = self.results_dir / "evaluation_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        # Also save complete evaluation info as JSON for reference
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
        print(f"- Evaluation metrics: {metrics_path}")
        print(f"- Evaluation info: {info_path}")
        
        return {
            'results_directory': str(self.results_dir),
            'detailed_results_path': str(detailed_path),
            'metrics_path': str(metrics_path),
            'info_path': str(info_path)
        }
    
    def _flatten_metrics_for_csv(self, metrics: Dict[str, Any], dataset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Flatten nested metrics dictionary for CSV format"""
        flattened = {
            'timestamp': self.timestamp,
            'dataset_name': self.dataset_name,
            'provider': self.provider,
            'model': self.model
        }
        
        # Add model configuration details using ModelConfigManager
        if self.model_config:
            model_csv_data = ModelConfigManager.format_for_csv(self.model_config)
            flattened.update(model_csv_data)
        
        # Add dataset info
        if dataset_info:
            flattened.update({
                'dataset_path': dataset_info.get('path', ''),
                'total_records_in_dataset': dataset_info.get('total_records', 0),
                'features_used': ', '.join(dataset_info.get('features', [])),
                'scam_count_in_dataset': dataset_info.get('scam_count', 0),
                'legitimate_count_in_dataset': dataset_info.get('legitimate_count', 0)
            })
        
        # Add main metrics
        if 'error' not in metrics:
            flattened.update({
                'total_samples_evaluated': metrics.get('total_samples_in_dataset', 0),
                'successfully_processed': metrics.get('successfully_processed', 0),
                'failed_predictions': metrics.get('failed_predictions', 0),
                'correct_predictions': metrics.get('correct_predictions', 0),
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'specificity': metrics.get('specificity', 0)
            })
            
            # Add confusion matrix
            cm = metrics.get('confusion_matrix', {})
            flattened.update({
                'true_positives': cm.get('true_positives', 0),
                'true_negatives': cm.get('true_negatives', 0),
                'false_positives': cm.get('false_positives', 0),
                'false_negatives': cm.get('false_negatives', 0)
            })
        else:
            flattened['error'] = metrics.get('error', 'Unknown error')
            
        return flattened
    
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