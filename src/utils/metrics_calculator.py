from typing import List, Dict, Any
import pandas as pd

class MetricsCalculator:
    """
    Calculate performance metrics for scam detection evaluation.
    """
    
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        self.valid_results = [r for r in results if r.get('predicted_label') is not None]
        
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.valid_results:
            return {'error': 'No valid results to calculate metrics'}
            
        total_samples = len(self.valid_results)
        correct_predictions = sum(1 for r in self.valid_results if r.get('is_correct', False))
        
        # Basic metrics
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        # Confusion matrix components
        tp = sum(1 for r in self.valid_results if r.get('actual_label') == 1 and r.get('predicted_label') == 1)
        tn = sum(1 for r in self.valid_results if r.get('actual_label') == 0 and r.get('predicted_label') == 0)
        fp = sum(1 for r in self.valid_results if r.get('actual_label') == 0 and r.get('predicted_label') == 1)
        fn = sum(1 for r in self.valid_results if r.get('actual_label') == 1 and r.get('predicted_label') == 0)
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Error analysis
        total_processed = len(self.results)
        failed_predictions = total_processed - total_samples
        
        metrics = {
            'total_samples_in_dataset': total_processed,
            'successfully_processed': total_samples,
            'failed_predictions': failed_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4),
            'specificity': round(specificity, 4),
            'confusion_matrix': {
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn
            }
        }
        
        return metrics
    
    def get_detailed_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown by categories"""
        breakdown = {
            'by_actual_label': {},
            'by_predicted_label': {},
            'error_analysis': []
        }
        
        # Breakdown by actual label
        for label in [0, 1]:
            label_results = [r for r in self.valid_results if r.get('actual_label') == label]
            if label_results:
                correct = sum(1 for r in label_results if r.get('is_correct', False))
                label_name = 'scam' if label == 1 else 'legitimate'
                breakdown['by_actual_label'][f'actual_{label_name}'] = {
                    'total': len(label_results),
                    'correct': correct,
                    'accuracy': round(correct / len(label_results), 4) if label_results else 0
                }
        
        # Breakdown by predicted label
        for label in [0, 1]:
            label_results = [r for r in self.valid_results if r.get('predicted_label') == label]
            if label_results:
                correct = sum(1 for r in label_results if r.get('is_correct', False))
                label_name = 'scam' if label == 1 else 'legitimate'
                breakdown['by_predicted_label'][f'predicted_{label_name}'] = {
                    'total': len(label_results),
                    'correct': correct,
                    'precision': round(correct / len(label_results), 4) if label_results else 0
                }
        
        # Error analysis
        error_results = [r for r in self.results if r.get('predicted_label') is None]
        for error_result in error_results:
            breakdown['error_analysis'].append({
                'id': error_result.get('id', 'Unknown'),
                'error_reason': error_result.get('llm_reason', 'Unknown error')
            })
        
        return breakdown
    
    def print_metrics_summary(self, dataset_info: Dict[str, Any] = None):
        """Print a formatted summary of metrics"""
        metrics = self.calculate_metrics()
        
        print("=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        if dataset_info:
            print(f"Dataset: {dataset_info.get('name', 'Unknown')}")
            print(f"Total records in dataset: {dataset_info.get('total_records', 'Unknown')}")
            print(f"Content features used: {', '.join(dataset_info.get('features', []))}")
            
            # Show original dataset distribution
            scam_count = dataset_info.get('scam_count', 0)
            legitimate_count = dataset_info.get('legitimate_count', 0)
            print(f"Original dataset distribution: {scam_count} scam, {legitimate_count} legitimate")
            
            # Show sample distribution (calculated from actual results)
            sample_scam = len([r for r in self.valid_results if r.get('actual_label') == 1])
            sample_legit = len([r for r in self.valid_results if r.get('actual_label') == 0])
            print(f"Sample distribution: {sample_scam} scam, {sample_legit} legitimate")
            print()
        
        if 'error' in metrics:
            print(f"Error: {metrics['error']}")
            return
            
        print(f"Successfully processed: {metrics['successfully_processed']} samples")
        print(f"Failed predictions: {metrics['failed_predictions']} samples")
        print(f"Overall accuracy: {metrics['accuracy']:.2%}")
        
        print("\nPERFORMANCE METRICS:")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall: {metrics['recall']:.2%}")
        print(f"F1 Score: {metrics['f1_score']:.2%}")
        print(f"Specificity: {metrics['specificity']:.2%}")
        
        cm = metrics['confusion_matrix']
        print(f"\nCONFUSION MATRIX:")
        print(f"True Positives (Scam correctly identified): {cm['true_positives']}")
        print(f"True Negatives (Legitimate correctly identified): {cm['true_negatives']}")
        print(f"False Positives (Legitimate misclassified as Scam): {cm['false_positives']}")
        print(f"False Negatives (Scam misclassified as Legitimate): {cm['false_negatives']}")
        
        print("=" * 80) 