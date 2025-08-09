import time
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.llm_core.api_provider import LLM
from src.llm_core.api_call import make_api_call, make_api_call_async
from src.utils.data_loader import DatasetLoader
from src.evaluate.prompt_generator import PromptGenerator
from src.utils.results_saver import ResultsSaver
import json
import asyncio

@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics for a single API call"""
    provider: str
    model: str
    response_time: float  # seconds
    success: bool
    prompt_category: str = "unknown"  # short, medium, long, dataset
    error_message: Optional[str] = None

class PerformanceEvaluator:
    """
    Evaluator class for measuring token/second performance across different LLM providers.
    """
    
    def __init__(self, 
                 dataset_path: Optional[str] = None,
                 sample_size: int = 50,
                 random_state: int = 42,
                 content_columns: Optional[List[str]] = None):
        """
        Initialize the performance evaluator
        
        Args:
            dataset_path: Path to dataset (optional - can use synthetic data if None)
            sample_size: Number of samples to test
            random_state: Random seed for reproducibility
            content_columns: Content columns to use (if using dataset)
        """
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.random_state = random_state
        self.content_columns = content_columns
        self.results = []
        
        # Initialize data loader if dataset provided
        if dataset_path:
            self.data_loader = DatasetLoader(dataset_path)
        else:
            self.data_loader = None
            

    
    def _create_synthetic_prompts(self) -> List[Dict[str, str]]:
        """Create synthetic test prompts of varying lengths"""
        prompts = []
        
        # Short prompts
        for i in range(self.sample_size // 3):
            prompts.append({
                "system": "You are a helpful assistant. Please respond concisely.",
                "user": f"What is {i + 1} + {i + 2}?",
                "category": "short"
            })
        
        # Medium prompts  
        for i in range(self.sample_size // 3):
            prompts.append({
                "system": "You are an expert analyst. Provide detailed analysis.",
                "user": f"Analyze the following scenario: A company has been experiencing declining sales for {i + 1} months. The main factors appear to be increased competition, changing customer preferences, and supply chain issues. What recommendations would you provide?",
                "category": "medium"
            })
        
        # Long prompts
        remaining = self.sample_size - len(prompts)
        for i in range(remaining):
            long_context = " ".join([f"This is sentence {j} in a long context that provides extensive background information." for j in range(50)])
            prompts.append({
                "system": "You are a comprehensive analysis assistant. Provide thorough responses based on the given context.",
                "user": f"Context: {long_context}\n\nBased on this context, please provide a detailed analysis of the implications and potential outcomes for scenario {i + 1}.",
                "category": "long"
            })
        
        return prompts
    
    def _prepare_prompts_from_dataset(self) -> List[Dict[str, str]]:
        """Prepare prompts from dataset"""
        if not self.data_loader:
            raise ValueError("No dataset loader available")
            
        # Load and sample data
        self.data_loader.load_dataset()
        sample_df = self.data_loader.sample_data(self.sample_size, self.random_state)
        
        # Initialize prompt generator
        if self.content_columns:
            missing_columns = [col for col in self.content_columns if col not in self.data_loader.features]
            if missing_columns:
                raise ValueError(f"Specified content columns not found: {missing_columns}")
            content_columns = self.content_columns
        else:
            content_columns = self.data_loader.features
            
        prompt_generator = PromptGenerator(self.data_loader.features, content_columns)
        system_prompt = prompt_generator.get_system_prompt()
        
        prompts = []
        for _, row in sample_df.iterrows():
            user_prompt = prompt_generator.create_user_prompt(row.to_dict())
            prompts.append({
                "system": system_prompt,
                "user": user_prompt,
                "category": "dataset"
            })
        
        return prompts
    
    def evaluate_model_performance(self, 
                                 provider: str, 
                                 model: str,
                                 prompts: Optional[List[Dict[str, str]]] = None) -> List[PerformanceMetrics]:
        """
        Evaluate performance for a single model
        
        Args:
            provider: LLM provider name
            model: Model name
            prompts: List of prompts (optional - will use synthetic if None)
            
        Returns:
            List of performance metrics
        """
        if prompts is None:
            if self.dataset_path:
                prompts = self._prepare_prompts_from_dataset()
            else:
                prompts = self._create_synthetic_prompts()
        
        print(f"\nEvaluating {provider} - {model}")
        print(f"Testing with {len(prompts)} prompts...")
        
        # Initialize LLM
        try:
            # Use Response API by default for OpenAI models
            use_response_api = (provider == "openai")
            llm_instance = LLM(provider=provider, model=model, use_response_api=use_response_api)
            llm = llm_instance.get_llm()
        except Exception as e:
            print(f"Failed to initialize {provider} - {model}: {e}")
            return []
        
        model_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"  Processing prompt {i+1}/{len(prompts)}...", end=" ")
            
            try:
                # Time the API call
                start_time = time.time()
                response = make_api_call(llm, prompt["system"], prompt["user"])
                end_time = time.time()
                
                response_time = end_time - start_time
                
                # Create performance metric
                metric = PerformanceMetrics(
                    provider=provider,
                    model=model,
                    response_time=response_time,
                    success=True,
                    prompt_category=prompt.get("category", "unknown")
                )
                
                model_results.append(metric)
                print(f"✓ {response_time:.2f}s")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                metric = PerformanceMetrics(
                    provider=provider,
                    model=model,
                    response_time=0,
                    success=False,
                    prompt_category=prompt.get("category", "unknown"),
                    error_message=str(e)
                )
                model_results.append(metric)
                
        return model_results

    async def evaluate_model_performance_async(self, 
                                             provider: str, 
                                             model: str,
                                             prompts: Optional[List[Dict[str, str]]] = None,
                                             concurrent_requests: int = 10) -> List[PerformanceMetrics]:
        """
        Evaluate performance for a single model asynchronously
        
        Args:
            provider: LLM provider name
            model: Model name
            prompts: List of prompts (optional - will use synthetic if None)
            concurrent_requests: Number of concurrent requests to make (default: 10)
            
        Returns:
            List of performance metrics
        """
        if prompts is None:
            if self.dataset_path:
                prompts = self._prepare_prompts_from_dataset()
            else:
                prompts = self._create_synthetic_prompts()
        
        print(f"\nEvaluating {provider} - {model} (Async)")
        print(f"Testing with {len(prompts)} prompts using {concurrent_requests} concurrent requests...")
        
        # Initialize LLM
        try:
            # Use Response API by default for OpenAI models
            use_response_api = (provider == "openai")
            llm_instance = LLM(provider=provider, model=model, use_response_api=use_response_api)
            llm = llm_instance.get_llm()
        except Exception as e:
            print(f"Failed to initialize {provider} - {model}: {e}")
            return []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def process_single_prompt(i: int, prompt: Dict[str, str]) -> PerformanceMetrics:
            async with semaphore:
                print(f"  Processing prompt {i+1}/{len(prompts)}...", end=" ")
                
                try:
                    # Time the API call
                    start_time = time.time()
                    response = await make_api_call_async(llm, prompt["system"], prompt["user"])
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    
                    # Create performance metric
                    metric = PerformanceMetrics(
                        provider=provider,
                        model=model,
                        response_time=response_time,
                        success=True,
                        prompt_category=prompt.get("category", "unknown")
                    )
                    
                    print(f"✓ {response_time:.2f}s")
                    return metric
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
                    metric = PerformanceMetrics(
                        provider=provider,
                        model=model,
                        response_time=0,
                        success=False,
                        prompt_category=prompt.get("category", "unknown"),
                        error_message=str(e)
                    )
                    return metric
        
        # Create tasks for all prompts
        tasks = []
        for i, prompt in enumerate(prompts):
            task = process_single_prompt(i, prompt)
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        model_results = await asyncio.gather(*tasks, return_exceptions=False)
        end_time = time.time()
        
        # Handle any exceptions that might have been returned
        valid_results = []
        for i, result in enumerate(model_results):
            if isinstance(result, Exception):
                print(f"Exception in prompt {i+1}: {result}")
                # Create error metric
                error_metric = PerformanceMetrics(
                    provider=provider,
                    model=model,
                    response_time=0,
                    success=False,
                    prompt_category="unknown",
                    error_message=str(result)
                )
                valid_results.append(error_metric)
            else:
                valid_results.append(result)
        
        total_time = end_time - start_time
        successful_results = [r for r in valid_results if r.success]
        print(f"\nAsync performance evaluation completed in {total_time:.2f} seconds.")
        print(f"Successful requests: {len(successful_results)}/{len(valid_results)}")
        if successful_results:
            avg_tokens_per_sec = sum(r.response_time for r in successful_results) / len(successful_results)
            print(f"Average response time: {avg_tokens_per_sec:.2f} seconds")
        
        return valid_results
    
    def evaluate_multiple_models(self, model_configs: List[Dict[str, str]]) -> Dict[str, List[PerformanceMetrics]]:
        """
        Evaluate performance across multiple models
        
        Args:
            model_configs: List of dicts with 'provider' and 'model' keys
            
        Returns:
            Dictionary mapping model names to performance metrics
        """
        all_results = {}
        
        # Prepare prompts once for all models
        if self.dataset_path:
            prompts = self._prepare_prompts_from_dataset()
        else:
            prompts = self._create_synthetic_prompts()
        
        print(f"="*80)
        print(f"PERFORMANCE EVALUATION - {len(model_configs)} MODELS")
        print(f"="*80)
        
        for config in model_configs:
            provider = config['provider']
            model = config['model']
            model_key = f"{provider}_{model}"
            
            results = self.evaluate_model_performance(provider, model, prompts)
            all_results[model_key] = results
            self.results.extend(results)
        
        return all_results
    
    def calculate_summary_statistics(self, results: Dict[str, List[PerformanceMetrics]]) -> pd.DataFrame:
        """Calculate summary statistics for all models"""
        summary_data = []
        
        for model_key, metrics_list in results.items():
            successful_metrics = [m for m in metrics_list if m.success]
            
            if not successful_metrics:
                summary_data.append({
                    'Model': model_key,
                    'Success_Rate': 0,
                    'Avg_Response_Time': 0,
                    'Min_Response_Time': 0,
                    'Max_Response_Time': 0,
                    'Total_Requests': len(metrics_list)
                })
                continue
            
            # Calculate statistics
            response_times = [m.response_time for m in successful_metrics]
            
            summary_data.append({
                'Model': model_key,
                'Success_Rate': len(successful_metrics) / len(metrics_list),
                'Avg_Response_Time': sum(response_times) / len(response_times),
                'Min_Response_Time': min(response_times),
                'Max_Response_Time': max(response_times),
                'Total_Requests': len(metrics_list)
            })
        
        return pd.DataFrame(summary_data)
    
    def print_performance_report(self, results: Dict[str, List[PerformanceMetrics]]):
        """Print a formatted performance report"""
        summary_df = self.calculate_summary_statistics(results)
        
        print(f"\n{'='*100}")
        print("PERFORMANCE EVALUATION REPORT")
        print(f"{'='*100}")
        
        # Sort by response time
        summary_df = summary_df.sort_values('Avg_Response_Time', ascending=True)
        
        print(f"{'Model':<25} {'Success%':<10} {'Avg Time':<10} {'Min Time':<10} {'Max Time':<10} {'Requests':<10}")
        print("-" * 100)
        
        for _, row in summary_df.iterrows():
            print(f"{row['Model']:<25} {row['Success_Rate']:<10.1%} {row['Avg_Response_Time']:<10.2f} "
                  f"{row['Min_Response_Time']:<10.2f} {row['Max_Response_Time']:<10.2f} {row['Total_Requests']:<10}")
        
        print(f"\n{'='*100}")
        
        # Best performing model
        if not summary_df.empty:
            best_model = summary_df.iloc[0]
            print(f"🏆 Best performing model: {best_model['Model']}")
            print(f"   Average response time: {best_model['Avg_Response_Time']:.2f} seconds")
            print(f"   Success rate: {best_model['Success_Rate']:.1%}")
    
    def save_results(self, results: Dict[str, List[PerformanceMetrics]], output_dir: str = "performance_results") -> Dict[str, str]:
        """Save performance results to files"""
        import os
        from datetime import datetime
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_results = []
        for model_key, metrics_list in results.items():
            for metric in metrics_list:
                detailed_results.append({
                    'model': model_key,
                    'provider': metric.provider,
                    'model_name': metric.model,
                    'response_time': metric.response_time,
                    'success': metric.success,
                    'prompt_category': metric.prompt_category,
                    'error_message': metric.error_message
                })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_path = os.path.join(output_dir, f"detailed_performance_{timestamp}.csv")
        detailed_df.to_csv(detailed_path, index=False)
        
        # Save summary statistics
        summary_df = self.calculate_summary_statistics(results)
        summary_path = os.path.join(output_dir, f"performance_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Save configuration info
        config_info = {
            'timestamp': timestamp,
            'sample_size': self.sample_size,
            'dataset_path': self.dataset_path,
            'content_columns': self.content_columns,
            'models_tested': list(results.keys())
        }
        
        config_path = os.path.join(output_dir, f"performance_config_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(config_info, f, indent=2)
        
        return {
            'detailed_results': detailed_path,
            'summary_results': summary_path,
            'config_file': config_path,
            'output_directory': output_dir
        } 