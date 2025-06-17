import pandas as pd
from typing import List, Dict, Any, Optional
from api_provider import LLM
from api_call import make_api_call
from data_loader import DatasetLoader
from prompt_generator import PromptGenerator
from metrics_calculator import MetricsCalculator
from results_saver import ResultsSaver

class ScamDetectionEvaluator:
    """
    Main evaluator class that orchestrates the entire scam detection evaluation pipeline.
    Designed to work with any dataset that has a 'label' column and text content.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 provider: str,
                 model: str,
                 sample_size: int = 100,
                 balanced_sample: bool = False,
                 random_state: int = 42,
                 content_columns: Optional[List[str]] = None):
        """
        Initialize the evaluator with dataset and model configuration
        
        Args:
            dataset_path: Path to the CSV dataset file
            provider: LLM provider (e.g., 'openai', 'anthropic', 'local')
            model: Model name
            sample_size: Number of samples to evaluate
            balanced_sample: Whether to sample equal numbers of scam and legitimate messages
            random_state: Random seed for reproducibility
            content_columns: List of column names to use as content for evaluation.
                           If None, uses all non-label columns.
        """
        self.dataset_path = dataset_path
        self.provider = provider
        self.model = model
        self.sample_size = sample_size
        self.balanced_sample = balanced_sample
        self.random_state = random_state
        self.content_columns = content_columns
        
        # Initialize components
        self.data_loader = DatasetLoader(dataset_path)
        self.llm_instance = None
        self.llm = None
        self.prompt_generator = None
        self.results = []
        
    def setup_llm(self):
        """Initialize the LLM"""
        try:
            self.llm_instance = LLM(provider=self.provider, model=self.model)
            self.llm = self.llm_instance.get_llm()
            print(f"LLM initialized successfully: {self.provider} - {self.model}")
        except Exception as e:
            raise Exception(f"Error initializing LLM: {e}")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load dataset and prepare sample for evaluation"""
        # Load full dataset
        self.data_loader.load_dataset()
        
        # Validate content columns if specified
        if self.content_columns:
            missing_columns = [col for col in self.content_columns if col not in self.data_loader.features]
            if missing_columns:
                raise ValueError(f"Specified content columns not found in dataset: {missing_columns}")
            print(f"Using specified content columns: {self.content_columns}")
        else:
            self.content_columns = self.data_loader.features
            print(f"Using all available features as content: {self.content_columns}")
        
        # Get sample (balanced or regular)
        if self.balanced_sample:
            sample_df = self.data_loader.sample_balanced_data(self.sample_size, self.random_state)
        else:
            sample_df = self.data_loader.sample_data(self.sample_size, self.random_state)
        
        # Initialize prompt generator with available features and specified content columns
        self.prompt_generator = PromptGenerator(self.data_loader.features, self.content_columns)
        
        print(f"\nDataset: {self.data_loader.dataset_name}")
        print(f"Sample size: {len(sample_df)} records")
        print(self.prompt_generator.get_features_summary())
        
        return sample_df
    
    def evaluate_sample(self, sample_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Evaluate the sample dataset using the LLM
        
        Args:
            sample_df: Sample dataframe to evaluate
            
        Returns:
            List of evaluation results
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Call setup_llm() first.")
        
        if self.prompt_generator is None:
            raise ValueError("Prompt generator not initialized. Call load_and_prepare_data() first.")
        
        results = []
        system_prompt = self.prompt_generator.get_system_prompt()
        
        print("\n" + "="*80)
        print("STARTING SCAM DETECTION EVALUATION")
        print("="*80)
        
        for i, (_, row) in enumerate(sample_df.iterrows()):
            print(f"\nEvaluating record {i+1}/{len(sample_df)}...")
            
            # Create user prompt with specified content features
            user_prompt = self.prompt_generator.create_user_prompt(row.to_dict())
            
            try:
                # Make API call
                response = make_api_call(self.llm, system_prompt, user_prompt)
                
                # Extract prediction
                predicted_scam = response.Phishing  # Note: API still uses "Phishing" key for compatibility
                predicted_label = 1 if predicted_scam else 0
                actual_label = row['label']
                
                # Calculate if prediction is correct
                is_correct = predicted_label == actual_label
                
                # Create comprehensive result record
                result = self._create_result_record(row, predicted_label, is_correct, response.Reason)
                
                results.append(result)
                
                print(f"  Actual: {'Scam' if actual_label == 1 else 'Legitimate'}")
                print(f"  Predicted: {'Scam' if predicted_label == 1 else 'Legitimate'}")
                print(f"  Correct: {is_correct}")
                print(f"  Reason: {response.Reason[:100]}...")
                
            except Exception as e:
                print(f"  Error processing record {i+1}: {e}")
                result = self._create_error_result_record(row, str(e))
                results.append(result)
        
        self.results = results
        print(f"\nEvaluation completed. Processed {len(results)} records.")
        return results
    
    def _create_result_record(self, 
                             row: pd.Series, 
                             predicted_label: int, 
                             is_correct: bool, 
                             llm_reason: str) -> Dict[str, Any]:
        """Create a comprehensive result record including original data"""
        result = {
            'actual_label': row['label'],
            'actual_class': 'Scam' if row['label'] == 1 else 'Legitimate',
            'predicted_label': predicted_label,
            'predicted_class': 'Scam' if predicted_label == 1 else 'Legitimate',
            'is_correct': is_correct,
            'llm_reason': llm_reason
        }
        
        # Add id column if it exists
        if 'id' in row:
            result['id'] = row['id']
        
        # Add all original features except id (to avoid duplication)
        for feature in self.data_loader.features:
            if feature != 'id':
                result[f'original_{feature}'] = row[feature]
        
        return result
    
    def _create_error_result_record(self, row: pd.Series, error_message: str) -> Dict[str, Any]:
        """Create an error result record"""
        result = {
            'actual_label': row['label'],
            'actual_class': 'Scam' if row['label'] == 1 else 'Legitimate',
            'predicted_label': None,
            'predicted_class': 'Error',
            'is_correct': False,
            'llm_reason': f'Error: {error_message}'
        }
        
        # Add id column if it exists
        if 'id' in row:
            result['id'] = row['id']
        
        # Add all original features except id (to avoid duplication)
        for feature in self.data_loader.features:
            if feature != 'id':
                result[f'original_{feature}'] = row[feature]
        
        return result
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        calculator = MetricsCalculator(self.results)
        metrics = calculator.calculate_metrics()
        
        # Print summary
        dataset_info = self.data_loader.get_dataset_info()
        dataset_info['features'] = self.content_columns
        calculator.print_metrics_summary(dataset_info)
        
        return metrics
    
    def save_results(self) -> Dict[str, str]:
        """Save results to the specified directory structure"""
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        dataset_info = self.data_loader.get_dataset_info()
        
        dataset_info['features'] = self.content_columns
        dataset_info['features_used'] = self.content_columns
        
        # Initialize results saver
        saver = ResultsSaver(
            dataset_name=self.data_loader.dataset_name,
            provider=self.provider,
            model=self.model
        )
        
        # Save results
        save_paths = saver.save_results(self.results, metrics, dataset_info)
        
        # Create summary report
        report_path = saver.create_summary_report(metrics, dataset_info)
        save_paths['summary_report_path'] = report_path
        
        return save_paths
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline: setup -> load -> evaluate -> calculate -> save
        
        Returns:
            Dictionary with evaluation results and file paths
        """
        print("="*80)
        print("SCAM DETECTION EVALUATION PIPELINE")
        print("="*80)
        
        # Setup LLM
        print("\n1. Setting up LLM...")
        self.setup_llm()
        
        # Load and prepare data
        print("\n2. Loading and preparing data...")
        sample_df = self.load_and_prepare_data()
        
        # Run evaluation
        print("\n3. Running evaluation...")
        results = self.evaluate_sample(sample_df)
        
        # Calculate metrics
        print("\n4. Calculating metrics...")
        metrics = self.calculate_metrics()
        
        # Save results
        print("\n5. Saving results...")
        save_paths = self.save_results()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return {
            'results': results,
            'metrics': metrics,
            'save_paths': save_paths,
            'dataset_info': self.data_loader.get_dataset_info()
        }

# Backward compatibility alias
PhishingEvaluator = ScamDetectionEvaluator 