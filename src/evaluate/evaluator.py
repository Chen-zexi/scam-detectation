import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.llm_core.api_provider import LLM
from src.llm_core.api_call import make_api_call, parse_structured_output, make_api_call_async, parse_structured_output_async, remove_thinking_tokens
from src.utils.data_loader import DatasetLoader
from src.evaluate.prompt_generator import PromptGenerator

# Evaluation-specific response schema
class EvaluationResponseSchema(BaseModel):
    Phishing: bool
    Reason: str
from src.utils.metrics_calculator import MetricsCalculator
from src.utils.results_saver import ResultsSaver
import asyncio
import time
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

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
                 enable_thinking: bool = False,
                 content_columns: Optional[List[str]] = None,
                 use_structure_model: bool = False):
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
        self.structure_model = None
        self.prompt_generator = None
        self.results = []
        self.use_structure_model = use_structure_model
        self.enable_thinking = enable_thinking
        
        # Checkpoint state
        self.current_index = 0
        self.checkpoint_file = None
        self.total_records = 0
        self.start_time = None
        self.last_checkpoint_message = None
        
    def setup_llm(self):
        """Initialize the LLM"""
        try:
            self.llm_instance = LLM(provider=self.provider, model=self.model)
            self.llm = self.llm_instance.get_llm()
            if self.use_structure_model:
                self.structure_model = self.llm_instance.get_structure_model()  # Uses gpt-4.1-nano by default
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
        
        # Calculate sample distribution
        sample_scam = len(sample_df[sample_df['label'] == 1])
        sample_legit = len(sample_df[sample_df['label'] == 0])
        
        print(f"\nDataset: {self.data_loader.dataset_name}")
        print(f"Sample size: {len(sample_df)} records")
        print(f"Sample distribution: {sample_scam} scam, {sample_legit} legitimate")
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
        
        # Use tqdm for progress tracking
        for i, (_, row) in enumerate(tqdm(sample_df.iterrows(), total=len(sample_df), desc="Evaluating")):
            
            # Create user prompt with specified content features
            user_prompt = self.prompt_generator.create_user_prompt(row.to_dict())
            
            try:
                if self.use_structure_model:
                    # Make API call
                    response = make_api_call(self.llm, system_prompt, user_prompt, response_schema=None, enable_thinking=self.enable_thinking, use_structure_model=True, structure_model=self.structure_model)
                    # Thinking tokens are automatically removed in make_api_call
                    response = parse_structured_output(self.structure_model, response, EvaluationResponseSchema)
                    # Response parsed successfully
                else:
                    # Make API call
                    response = make_api_call(self.llm, system_prompt, user_prompt, response_schema=EvaluationResponseSchema, use_structure_model=False)
                
                # Extract prediction
                predicted_scam = response.Phishing  # Note: API still uses "Phishing" key for compatibility
                predicted_label = 1 if predicted_scam else 0
                actual_label = int(row['label'])  # Should be safe since data is pre-cleaned
                
                # Calculate if prediction is correct
                is_correct = predicted_label == actual_label
                
                # Create comprehensive result record
                result = self._create_result_record(row, predicted_label, is_correct, response.Reason)
                
                results.append(result)
                
                # Use tqdm.write to avoid interfering with progress bar
                tqdm.write(f"  Record {i+1}: Actual={'Scam' if actual_label == 1 else 'Legitimate'}, "
                          f"Predicted={'Scam' if predicted_label == 1 else 'Legitimate'}, "
                          f"Correct={is_correct}")
                
            except Exception as e:
                tqdm.write(f"  Error processing record {i+1}: {e}")
                result = self._create_error_result_record(row, str(e))
                results.append(result)
        
        self.results = results
        print(f"\nEvaluation completed. Processed {len(results)} records.")
        return results

    async def evaluate_sample_async(self, sample_df: pd.DataFrame, concurrent_requests: int = 10) -> List[Dict[str, Any]]:
        """
        Evaluate the sample dataset using the LLM with concurrent requests
        
        Args:
            sample_df: Sample dataframe to evaluate
            concurrent_requests: Number of concurrent requests to make (default: 10)
            
        Returns:
            List of evaluation results
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Call setup_llm() first.")
        
        if self.prompt_generator is None:
            raise ValueError("Prompt generator not initialized. Call load_and_prepare_data() first.")
        
        system_prompt = self.prompt_generator.get_system_prompt()
        
        print("\n" + "="*80)
        print("STARTING SCAM DETECTION EVALUATION (ASYNC)")
        print(f"Concurrent requests: {concurrent_requests}")
        print("="*80)
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        # Create progress bar for async processing
        pbar = tqdm(total=len(sample_df), desc="Evaluating (async)")
        
        async def evaluate_single_record(i: int, row: pd.Series) -> Dict[str, Any]:
            async with semaphore:
                # Update progress bar description
                pbar.set_description(f"Evaluating record {i+1}/{len(sample_df)}")
                
                # Create user prompt with specified content features
                user_prompt = self.prompt_generator.create_user_prompt(row.to_dict())
                
                try:
                    if self.use_structure_model:
                        # Make async API call
                        response = await make_api_call_async(self.llm, system_prompt, user_prompt, 
                                                           response_schema=None, enable_thinking=self.enable_thinking, use_structure_model=True, structure_model=self.structure_model)
                        # Thinking tokens are automatically removed in make_api_call_async
                        response = await parse_structured_output_async(self.structure_model, response, EvaluationResponseSchema)
                        # Response parsed successfully
                    else:
                        # Make async API call
                        response = await make_api_call_async(self.llm, system_prompt, user_prompt, response_schema=EvaluationResponseSchema, use_structure_model=False)
                    
                    # Extract prediction
                    predicted_scam = response.Phishing  # Note: API still uses "Phishing" key for compatibility
                    predicted_label = 1 if predicted_scam else 0
                    actual_label = int(row['label'])  # Should be safe since data is pre-cleaned
                    
                    # Calculate if prediction is correct
                    is_correct = predicted_label == actual_label
                    
                    # Create comprehensive result record
                    result = self._create_result_record(row, predicted_label, is_correct, response.Reason)
                    
                    tqdm.write(f"Record {i+1} - Actual: {'Scam' if actual_label == 1 else 'Legitimate'}, "
                              f"Predicted: {'Scam' if predicted_label == 1 else 'Legitimate'}, Correct: {is_correct}")
                    
                    return result
                    
                except Exception as e:
                    tqdm.write(f"  Error processing record {i+1}: {e}")
                    result = self._create_error_result_record(row, str(e))
                    return result
                finally:
                    # Update progress bar after each record
                    pbar.update(1)
        
        # Create tasks for all records
        tasks = []
        for i, (_, row) in enumerate(sample_df.iterrows()):
            task = evaluate_single_record(i, row)
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=False)
        end_time = time.time()
        
        # Close progress bar
        pbar.close()
        
        # Handle any exceptions that might have been returned
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tqdm.write(f"Exception in record {i+1}: {result}")
                # Create error record
                row = sample_df.iloc[i]
                error_result = self._create_error_result_record(row, str(result))
                valid_results.append(error_result)
            else:
                valid_results.append(result)
        
        self.results = valid_results
        total_time = end_time - start_time
        print(f"\nAsync evaluation completed in {total_time:.2f} seconds. Processed {len(valid_results)} records.")
        print(f"Average time per record: {total_time/len(valid_results):.2f} seconds")
        return valid_results
    
    def _create_result_record(self, 
                             row: pd.Series, 
                             predicted_label: int, 
                             is_correct: bool, 
                             llm_reason: str) -> Dict[str, Any]:
        """Create a comprehensive result record including original data"""
        # Label should be clean at this point due to pre-filtering
        actual_label = int(row['label'])
        
        result = {
            'actual_label': actual_label,
            'actual_class': 'Scam' if actual_label == 1 else 'Legitimate',
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
        # Label should be clean at this point due to pre-filtering
        actual_label = int(row['label'])
        
        result = {
            'actual_label': actual_label,
            'actual_class': 'Scam' if actual_label == 1 else 'Legitimate',
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
        #calculator.print_metrics_summary(dataset_info)
        
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
        calculator = MetricsCalculator(self.results)
        calculator.print_metrics_summary(dataset_info)
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

    async def run_full_evaluation_async(self, concurrent_requests: int = 10) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline asynchronously: setup -> load -> evaluate -> calculate -> save
        
        Args:
            concurrent_requests: Number of concurrent requests to make (default: 10)
            
        Returns:
            Dictionary with evaluation results and file paths
        """
        print("="*80)
        print("SCAM DETECTION EVALUATION PIPELINE (ASYNC)")
        print("="*80)
        
        # Setup LLM
        print("\n1. Setting up LLM...")
        self.setup_llm()
        
        # Load and prepare data
        print("\n2. Loading and preparing data...")
        sample_df = self.load_and_prepare_data()
        
        # Run evaluation asynchronously
        print("\n3. Running evaluation asynchronously...")
        results = await self.evaluate_sample_async(sample_df, concurrent_requests)
        
        # Calculate metrics
        print("\n4. Calculating metrics...")
        metrics = self.calculate_metrics()
        
        # Save results
        print("\n5. Saving results...")
        save_paths = self.save_results()
        
        return {
            'results': results,
            'metrics': metrics,
            'save_paths': save_paths,
            'dataset_info': self.data_loader.get_dataset_info()
        }
    
    # ==================== CHECKPOINT FUNCTIONALITY ====================
    
    def _get_checkpoint_filename(self) -> str:
        """Generate checkpoint filename based on configuration"""
        dataset_name = Path(self.dataset_path).stem
        # Sanitize model name for filesystem compatibility
        safe_model_name = self.model.replace("/", "-").replace("\\", "-").replace(":", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{dataset_name}_evaluation_{self.provider}_{safe_model_name}_{timestamp}.json"
    
    def _find_existing_checkpoint(self, checkpoint_dir: str = "checkpoints") -> Optional[Path]:
        """Find the most recent checkpoint file for this configuration"""
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None
            
        dataset_name = Path(self.dataset_path).stem
        # Sanitize model name for filesystem compatibility
        safe_model_name = self.model.replace("/", "-").replace("\\", "-").replace(":", "-")
        pattern = f"{dataset_name}_evaluation_{self.provider}_{safe_model_name}_*.json"
        
        checkpoint_files = list(checkpoint_path.glob(pattern))
        if checkpoint_files:
            return max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        return None
    
    def load_checkpoint(self, checkpoint_dir: str = "checkpoints", 
                       resume_from_checkpoint: bool = True,
                       override_compatibility: bool = False) -> bool:
        """Load existing checkpoint if available and resume_from_checkpoint is True"""
        if not resume_from_checkpoint:
            return False
            
        checkpoint_file = self._find_existing_checkpoint(checkpoint_dir)
        if not checkpoint_file:
            print("No existing checkpoint found. Starting from beginning.")
            return False
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Validate checkpoint compatibility
            if not override_compatibility and (checkpoint_data.get('dataset_path') != self.dataset_path or
                checkpoint_data.get('provider') != self.provider or
                checkpoint_data.get('model') != self.model):
                print("Checkpoint configuration mismatch. Starting from beginning.")
                return False
            
            # Show compatibility override message if needed
            if override_compatibility and (checkpoint_data.get('dataset_path') != self.dataset_path or
                checkpoint_data.get('provider') != self.provider or
                checkpoint_data.get('model') != self.model):
                print("Loading checkpoint with compatibility override:")
                print(f"  Original provider: {checkpoint_data.get('provider')} -> Current: {self.provider}")
                print(f"  Original model: {checkpoint_data.get('model')} -> Current: {self.model}")
            
            # Load checkpoint state
            self.current_index = checkpoint_data.get('current_index', 0)
            self.results = checkpoint_data.get('results', [])
            self.checkpoint_file = checkpoint_file
            
            print(f"Loaded checkpoint from {checkpoint_file}")
            print(f"Resuming from index {self.current_index} with {len(self.results)} existing results")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from beginning.")
            return False
    
    def save_checkpoint(self, checkpoint_dir: str = "checkpoints"):
        """Save current processing state to checkpoint file"""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        if not self.checkpoint_file:
            self.checkpoint_file = checkpoint_path / self._get_checkpoint_filename()
        
        checkpoint_data = {
            'dataset_path': self.dataset_path,
            'provider': self.provider,
            'model': self.model,
            'sample_size': self.sample_size,
            'content_columns': self.content_columns,
            'enable_thinking': self.enable_thinking,
            'use_structure_model': self.use_structure_model,
            'current_index': self.current_index,
            'total_records': self.total_records,
            'results': self.results,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def _write_checkpoint_message(self, message: str, progress_bar=None):
        """Write checkpoint message using tqdm-friendly approach"""
        # Use tqdm.write() which automatically positions the message correctly
        tqdm.write(f"📁 {message}")
        self.last_checkpoint_message = message
        
        # If we have a progress bar reference, update its postfix
        if progress_bar:
            progress_bar.set_postfix_str("✓ Checkpoint saved")
            # Clear the postfix after a brief moment to keep it clean
            progress_bar.refresh()
    
    def process_full_dataset_with_checkpoints(self, 
                                            checkpoint_interval: int = 1000,
                                            checkpoint_dir: str = "checkpoints",
                                            resume_from_checkpoint: bool = True,
                                            override_compatibility: bool = False) -> Dict[str, Any]:
        """
        Process the entire dataset (no sampling) with checkpointing capabilities
        
        Args:
            checkpoint_interval: Number of records to process before saving checkpoint
            checkpoint_dir: Directory to save checkpoints
            resume_from_checkpoint: Whether to resume from existing checkpoint if found
            override_compatibility: Whether to override checkpoint compatibility checks
            
        Returns:
            Dictionary with evaluation results and file paths
        """
        print("="*80)
        print("EVALUATION PIPELINE WITH CHECKPOINTING")
        print("="*80)
        
        # Setup LLM
        self.setup_llm()
        
        # Load full dataset (no sampling)
        self.data_loader.load_dataset()
        dataset_df = self.data_loader.df
        self.total_records = len(dataset_df)
        
        # Validate content columns
        if self.content_columns:
            missing_columns = [col for col in self.content_columns if col not in self.data_loader.features]
            if missing_columns:
                raise ValueError(f"Specified content columns not found: {missing_columns}")
        else:
            self.content_columns = self.data_loader.features
        
        # Initialize prompt generator
        self.prompt_generator = PromptGenerator(self.data_loader.features, self.content_columns)
        
        print(f"Dataset: {self.data_loader.dataset_name}")
        print(f"Total records: {self.total_records:,}")
        print(f"Checkpoint interval: {checkpoint_interval:,}")
        print(f"Content features: {', '.join(self.content_columns)}")
        
        # Load checkpoint if available
        self.load_checkpoint(checkpoint_dir, resume_from_checkpoint, override_compatibility)
        
        self.start_time = time.time()
        total_processed = len(self.results)
        
        print(f"\nProcessing records {self.current_index + 1} to {self.total_records}...")
        
        # Get system prompt
        system_prompt = self.prompt_generator.get_system_prompt()
        
        # Create progress bar
        progress_bar = tqdm(
            range(self.current_index, self.total_records),
            desc="Evaluating",
            unit="records",
            initial=self.current_index,
            total=self.total_records
        )
        
        # Process remaining records
        for i in progress_bar:
            row = dataset_df.iloc[i]
            
            # Update progress bar description
            progress_bar.set_description(f"Evaluating record {i + 1}/{self.total_records}")
            
            try:
                # Create user prompt
                user_prompt = self.prompt_generator.create_user_prompt(row.to_dict())
                
                # Make API call
                if self.use_structure_model:
                    response = make_api_call(self.llm, system_prompt, user_prompt,
                                           response_schema=None, enable_thinking=self.enable_thinking, use_structure_model=True, structure_model=self.structure_model)
                    # Thinking tokens are automatically removed in make_api_call
                    response = parse_structured_output(self.structure_model, response, EvaluationResponseSchema)
                else:
                    response = make_api_call(self.llm, system_prompt, user_prompt, response_schema=EvaluationResponseSchema, use_structure_model=False)
                
                # Extract prediction
                predicted_scam = response.Phishing
                predicted_label = 1 if predicted_scam else 0
                actual_label = row['label']
                is_correct = predicted_label == actual_label
                
                # Create evaluation record
                result = self._create_result_record(row, predicted_label, is_correct, response.Reason)
                
            except Exception as e:
                tqdm.write(f"Error processing record {i + 1}: {e}")
                result = self._create_error_result_record(row, str(e))
            
            self.results.append(result)
            self.current_index = i + 1
            total_processed += 1
            
            # Save checkpoint at intervals
            if (i + 1) % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir)
                self._write_checkpoint_message(f"Checkpoint saved at record {i + 1}", progress_bar)
        
        # Close progress bar
        progress_bar.close()
        
        # Final checkpoint
        self.save_checkpoint(checkpoint_dir)
        
        # Calculate metrics and save final results
        metrics = self.calculate_metrics()
        save_paths = self.save_results()
        
        total_time = time.time() - self.start_time
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETED")
        print(f"{'='*80}")
        print(f"Total records processed: {total_processed:,}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per record: {total_time/total_processed:.3f} seconds")
        print(f"Accuracy: {metrics.get('accuracy', 0):.2%}")
        
        return {
            'results': self.results,
            'metrics': metrics,
            'save_paths': save_paths,
            'dataset_info': self.data_loader.get_dataset_info(),
            'summary': {
                'total_records': len(self.results),
                'successful_evaluations': len([r for r in self.results if r.get('predicted_label') is not None]),
                'accuracy': metrics.get('accuracy', 0),
                'checkpoint_file': str(self.checkpoint_file) if self.checkpoint_file else None
            }
        }
    
    async def process_full_dataset_with_checkpoints_async_chunked(self, 
                                                                 checkpoint_interval: int = 1000,
                                                                 checkpoint_dir: str = "checkpoints",
                                                                 resume_from_checkpoint: bool = True,
                                                                 concurrent_requests: int = 10,
                                                                 override_compatibility: bool = False) -> Dict[str, Any]:
        """
        Process the entire dataset using chunked batching (legacy method)
        
        Args:
            checkpoint_interval: Number of records to process before saving checkpoint
            checkpoint_dir: Directory to save checkpoints  
            resume_from_checkpoint: Whether to resume from existing checkpoint if found
            concurrent_requests: Number of concurrent requests to make
            override_compatibility: Whether to override checkpoint compatibility checks
            
        Returns:
            Dictionary with evaluation results and file paths
        """
        print("="*80)
        print("ASYNC EVALUATION PIPELINE WITH CHUNKED BATCHING")
        print(f"Concurrent requests: {concurrent_requests}")
        print("="*80)
        
        # Setup LLM
        self.setup_llm()
        
        # Load full dataset (no sampling)
        self.data_loader.load_dataset()
        dataset_df = self.data_loader.df
        self.total_records = len(dataset_df)
        
        # Validate content columns
        if self.content_columns:
            missing_columns = [col for col in self.content_columns if col not in self.data_loader.features]
            if missing_columns:
                raise ValueError(f"Specified content columns not found: {missing_columns}")
        else:
            self.content_columns = self.data_loader.features
        
        # Initialize prompt generator
        self.prompt_generator = PromptGenerator(self.data_loader.features, self.content_columns)
        
        print(f"Dataset: {self.data_loader.dataset_name}")
        print(f"Total records: {self.total_records:,}")
        print(f"Checkpoint interval: {checkpoint_interval:,}")
        print(f"Content features: {', '.join(self.content_columns)}")
        
        # Load checkpoint if available
        self.load_checkpoint(checkpoint_dir, resume_from_checkpoint, override_compatibility)
        
        self.start_time = time.time()
        
        print(f"\nProcessing records {self.current_index + 1} to {self.total_records}...")
        
        # Process in chunks to manage memory and checkpointing
        chunk_size = min(checkpoint_interval, concurrent_requests * 10)
        
        # Create overall progress bar
        overall_progress = tqdm(
            total=self.total_records,
            desc="Overall Progress",
            unit="records",
            initial=self.current_index,
            position=0,
            leave=False
        )
        
        for chunk_start in range(self.current_index, self.total_records, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.total_records)
            chunk_size_actual = chunk_end - chunk_start
            
            # Create chunk progress bar
            chunk_progress = tqdm(
                total=chunk_size_actual,
                desc=f"Chunk {chunk_start + 1}-{chunk_end}",
                unit="records",
                position=1,
                leave=False
            )
            
            # Create tasks for this chunk
            async def evaluate_with_semaphore(index: int, row: pd.Series):
                async with semaphore:
                    result = await self._evaluate_single_record_async(index, row)
                    chunk_progress.update(1)
                    return result
            
            tasks = []
            for i in range(chunk_start, chunk_end):
                row = dataset_df.iloc[i]
                tasks.append(evaluate_with_semaphore(i, row))
            
            # Process chunk
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for i, result in enumerate(chunk_results):
                actual_index = chunk_start + i
                if isinstance(result, Exception):
                    row = dataset_df.iloc[actual_index]
                    result = self._create_error_result_record(row, str(result))
                
                self.results.append(result)
                self.current_index = actual_index + 1
                overall_progress.update(1)
            
            # Close chunk progress bar
            chunk_progress.close()
            
            # Save checkpoint after each chunk
            if chunk_end % checkpoint_interval <= chunk_size:
                self.save_checkpoint(checkpoint_dir)
                self._write_checkpoint_message(f"Checkpoint saved at record {chunk_end}", overall_progress)
        
        # Close overall progress bar
        overall_progress.close()
        
        # Final checkpoint
        self.save_checkpoint(checkpoint_dir)
        
        # Calculate metrics and save final results
        metrics = self.calculate_metrics()
        save_paths = self.save_results()
        
        total_time = time.time() - self.start_time
        total_processed = len(self.results)
        
        print(f"\n{'='*80}")
        print("CHUNKED ASYNC EVALUATION COMPLETED")
        print(f"{'='*80}")
        print(f"Total records processed: {total_processed:,}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per record: {total_time/total_processed:.3f} seconds")
        print(f"Accuracy: {metrics.get('accuracy', 0):.2%}")
        
        return {
            'results': self.results,
            'metrics': metrics,
            'save_paths': save_paths,
            'dataset_info': self.data_loader.get_dataset_info(),
            'summary': {
                'total_records': len(self.results),
                'successful_evaluations': len([r for r in self.results if r.get('predicted_label') is not None]),
                'accuracy': metrics.get('accuracy', 0),
                'checkpoint_file': str(self.checkpoint_file) if self.checkpoint_file else None
            }
        }
    
    async def _evaluate_single_record_async(self, index: int, row: pd.Series) -> Dict[str, Any]:
        """Evaluate a single record asynchronously"""
        try:
            system_prompt = self.prompt_generator.get_system_prompt()
            user_prompt = self.prompt_generator.create_user_prompt(row.to_dict())
            
            if self.use_structure_model:
                response = await make_api_call_async(self.llm, system_prompt, user_prompt,
                                                   response_schema=None, enable_thinking=self.enable_thinking, use_structure_model=True, structure_model=self.structure_model)
                # Thinking tokens are automatically removed in make_api_call_async
                response = await parse_structured_output_async(self.structure_model, response, EvaluationResponseSchema)
            else:
                response = await make_api_call_async(self.llm, system_prompt, user_prompt, response_schema=EvaluationResponseSchema, use_structure_model=False)
            
            predicted_scam = response.Phishing
            predicted_label = 1 if predicted_scam else 0
            actual_label = row['label']
            is_correct = predicted_label == actual_label
            
            return self._create_result_record(row, predicted_label, is_correct, response.Reason)
            
        except Exception as e:
            return self._create_error_result_record(row, str(e))

    async def process_full_dataset_with_checkpoints_async(self, 
                                                         checkpoint_interval: int = 1000,
                                                         checkpoint_dir: str = "checkpoints",
                                                         resume_from_checkpoint: bool = True,
                                                         concurrent_requests: int = 10,
                                                         override_compatibility: bool = False) -> Dict[str, Any]:
        """
        Process the entire dataset with asynchronous overlapping batches
        
        This method starts new requests as soon as slots become available rather than 
        waiting for entire chunks to complete, eliminating convoy effects from slow requests.
        
        Args:
            checkpoint_interval: Number of records to process before saving checkpoint
            checkpoint_dir: Directory to save checkpoints  
            resume_from_checkpoint: Whether to resume from existing checkpoint if found
            concurrent_requests: Number of concurrent requests to maintain
            override_compatibility: Whether to override checkpoint compatibility checks
            
        Returns:
            Dictionary with evaluation results and file paths
        """
        print("="*80)
        print("ASYNC EVALUATION PIPELINE WITH OVERLAPPING BATCHES")
        print(f"Concurrent requests: {concurrent_requests}")
        print("="*80)
        
        # Setup LLM
        self.setup_llm()
        
        # Load full dataset (no sampling)
        self.data_loader.load_dataset()
        dataset_df = self.data_loader.df
        self.total_records = len(dataset_df)
        
        # Validate content columns
        if self.content_columns:
            missing_columns = [col for col in self.content_columns if col not in self.data_loader.features]
            if missing_columns:
                raise ValueError(f"Specified content columns not found: {missing_columns}")
        else:
            self.content_columns = self.data_loader.features
        
        # Initialize prompt generator
        self.prompt_generator = PromptGenerator(self.data_loader.features, self.content_columns)
        
        print(f"Dataset: {self.data_loader.dataset_name}")
        print(f"Total records: {self.total_records:,}")
        print(f"Checkpoint interval: {checkpoint_interval:,}")
        print(f"Content features: {', '.join(self.content_columns)}")
        
        # Load checkpoint if available
        self.load_checkpoint(checkpoint_dir, resume_from_checkpoint, override_compatibility)
        
        self.start_time = time.time()
        
        print(f"\nProcessing records {self.current_index + 1} to {self.total_records}...")
        
        # Show next checkpoint info
        next_checkpoint = ((self.current_index // checkpoint_interval) + 1) * checkpoint_interval
        if next_checkpoint <= self.total_records:
            records_until_checkpoint = next_checkpoint - self.current_index
            print(f"Next checkpoint at record {next_checkpoint:,} ({records_until_checkpoint:,} records away)")
        
        # Create overall progress bar with enhanced statistics
        overall_progress = tqdm(
            total=self.total_records,
            desc="Overall Progress",
            unit="records",
            initial=self.current_index,
            position=0,
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Enhanced time tracking
        last_time_update = time.time()
        last_update_index = self.current_index
        rates_history = []  # Track rate history for smoothing
        session_start_index = self.current_index  # Track starting point for current session
        
        # Initialize overlapping batch processor
        completed_results = {}  # index -> result
        active_tasks = {}  # task -> index
        next_index = self.current_index
        
        # Create semaphore for concurrent control
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def process_single_record(index: int, row: pd.Series):
            """Process a single record with semaphore control"""
            async with semaphore:
                return await self._evaluate_single_record_async(index, row)
        
        # Main processing loop with overlapping batches
        while next_index < self.total_records or active_tasks:
            
            # Start new tasks up to concurrent limit
            while len(active_tasks) < concurrent_requests and next_index < self.total_records:
                row = dataset_df.iloc[next_index]
                task = asyncio.create_task(process_single_record(next_index, row))
                active_tasks[task] = next_index
                next_index += 1
            
            # Wait for at least one task to complete
            if active_tasks:
                done, pending = await asyncio.wait(
                    active_tasks.keys(), 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task in done:
                    index = active_tasks.pop(task)
                    try:
                        result = await task
                    except Exception as e:
                        row = dataset_df.iloc[index]
                        result = self._create_error_result_record(row, str(e))
                    
                    completed_results[index] = result
                    overall_progress.update(1)
                
                # Process completed results in order and add to final results
                while self.current_index in completed_results:
                    result = completed_results.pop(self.current_index)
                    self.results.append(result)
                    self.current_index += 1
                    
                    # Enhanced time estimation updates every 100 records
                    if self.current_index % 100 == 0:
                        current_time = time.time()
                        time_diff = current_time - last_time_update
                        records_diff = self.current_index - last_update_index
                        
                        # Only update if we have reasonable time difference (at least 1 second)
                        if time_diff >= 1.0 and records_diff > 0:
                            current_rate = records_diff / time_diff
                            
                            # Sanity check: rate should be reasonable (0.1 to 100 records/second)
                            if 0.1 <= current_rate <= 100.0:
                                rates_history.append(current_rate)
                                
                                # Keep only recent rates for smoothing (last 10 measurements)
                                if len(rates_history) > 10:
                                    rates_history = rates_history[-10:]
                                
                                # Calculate smoothed rate
                                avg_rate = sum(rates_history) / len(rates_history)
                                remaining_records = self.total_records - self.current_index
                                remaining_time = remaining_records / avg_rate if avg_rate > 0 else 0
                                
                                # Update tqdm description with enhanced info
                                elapsed_total = current_time - self.start_time
                                records_processed_this_session = self.current_index - session_start_index
                                overall_rate = records_processed_this_session / elapsed_total if elapsed_total > 0 else 0
                                
                                # Format remaining time nicely
                                if remaining_time > 3600:
                                    time_str = f"{remaining_time/3600:.1f}h"
                                elif remaining_time > 60:
                                    time_str = f"{remaining_time/60:.1f}m"
                                else:
                                    time_str = f"{remaining_time:.0f}s"
                                
                                overall_progress.set_description(
                                    f"Progress (Current: {current_rate:.2f}r/s, Avg: {overall_rate:.2f}r/s, ETA: {time_str})"
                                )
                                
                                last_time_update = current_time
                                last_update_index = self.current_index
                        
                        # Fallback: if no valid current rate, still update with overall average
                        elif time_diff >= 1.0:
                            elapsed_total = current_time - self.start_time
                            records_processed_this_session = self.current_index - session_start_index
                            overall_rate = records_processed_this_session / elapsed_total if elapsed_total > 0 else 0
                            
                            if rates_history:
                                avg_rate = sum(rates_history) / len(rates_history)
                                remaining_records = self.total_records - self.current_index
                                remaining_time = remaining_records / avg_rate
                                
                                if remaining_time > 3600:
                                    time_str = f"{remaining_time/3600:.1f}h"
                                elif remaining_time > 60:
                                    time_str = f"{remaining_time/60:.1f}m"
                                else:
                                    time_str = f"{remaining_time:.0f}s"
                                
                                overall_progress.set_description(
                                    f"Progress (Avg: {overall_rate:.2f}r/s, ETA: {time_str})"
                                )
                            else:
                                overall_progress.set_description(f"Progress (Avg: {overall_rate:.2f}r/s)")
                            
                            last_time_update = current_time
                            last_update_index = self.current_index
                    
                    # Save checkpoint periodically at fixed intervals (like sync version)
                    if self.current_index % checkpoint_interval == 0:
                        self.save_checkpoint(checkpoint_dir)
                        
                        # Calculate session progress for checkpoint messaging
                        records_processed_this_session = self.current_index - session_start_index
                        
                        # Update checkpoint with enhanced timing data
                        try:
                            checkpoint_file = Path(checkpoint_dir) / f"{self.checkpoint_file}.json"
                            if checkpoint_file.exists():
                                with open(checkpoint_file, 'r+') as f:
                                    checkpoint_data = json.load(f)
                                    f.seek(0)
                                    
                                    # Add enhanced timing information
                                    elapsed_time = time.time() - self.start_time
                                    checkpoint_data['elapsed_time'] = elapsed_time
                                    checkpoint_data['current_rate'] = records_processed_this_session / elapsed_time if elapsed_time > 0 else 0
                                    checkpoint_data['recent_rates'] = rates_history[-5:]  # Last 5 rates
                                    
                                    json.dump(checkpoint_data, f, indent=2)
                                    f.truncate()
                        except Exception as e:
                            print(f"Warning: Could not update checkpoint timing data: {e}")
                        
                        elapsed_session = time.time() - self.start_time
                        self._write_checkpoint_message(
                            f"✓ Checkpoint saved at record {self.current_index:,} "
                            f"(session: {elapsed_session/60:.1f}m, rate: {records_processed_this_session/elapsed_session:.2f}r/s)",
                            overall_progress
                        )
        
        # Close progress bar
        overall_progress.close()
        
        # Final enhanced summary
        total_time = time.time() - self.start_time
        records_processed_this_session = self.current_index - session_start_index
        final_rate = records_processed_this_session / total_time if total_time > 0 else 0
        
        print(f"\nFINAL TIMING STATISTICS (Current Session)")
        print(f"   Session time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        print(f"   Session rate: {final_rate:.2f} records/second")
        print(f"   Records processed this session: {records_processed_this_session:,}")
        print(f"   Total records completed: {len(self.results):,}")
        
        if rates_history:
            print(f"   Recent rate: {rates_history[-1]:.2f} records/second")
            print(f"   Rate stability: {min(rates_history)/max(rates_history):.2f} (1.0 = perfectly stable)")
        
        # Final checkpoint
        self.save_checkpoint(checkpoint_dir)
        
        # Calculate metrics and save final results
        metrics = self.calculate_metrics()
        save_paths = self.save_results()
        
        print(f"\n{'='*80}")
        print("ASYNC EVALUATION COMPLETED")
        print(f"{'='*80}")
        print(f"Total records completed: {len(self.results):,}")
        print(f"Session processing rate: {final_rate:.2f} records/second")
        print(f"Accuracy: {metrics.get('accuracy', 0):.2%}")
        
        return {
            'results': self.results,
            'metrics': metrics,
            'save_paths': save_paths,
            'dataset_info': self.data_loader.get_dataset_info(),
            'summary': {
                'total_records': len(self.results),
                'successful_evaluations': len([r for r in self.results if r.get('predicted_label') is not None]),
                'correct_predictions': len([r for r in self.results if r.get('is_correct', False)]),
                'accuracy': metrics.get('accuracy', 0),
                'success_rate': len([r for r in self.results if r.get('predicted_label') is not None]) / len(self.results),
                'final_processing_rate': final_rate,
                'total_processing_time': total_time,
                'checkpoint_file': str(self.checkpoint_file) if self.checkpoint_file else None
            }
        }

# Backward compatibility alias
PhishingEvaluator = ScamDetectionEvaluator 