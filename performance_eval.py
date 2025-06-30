#!/usr/bin/env python3
"""
Example usage of the Performance Evaluation Pipeline with Async Support

This script demonstrates different ways to use the performance evaluation pipeline
to measure token/second performance across different models and providers using
concurrent requests for faster testing.
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.performance_evaluator import PerformanceEvaluator

async def performance_evaluation_async():
    """
    Async performance evaluation with concurrent requests
    """
    print("\n" + "="*80)
    print("ASYNC PERFORMANCE EVALUATION")
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
        print("No dataset found, using synthetic prompts for testing.")
    
    model_configs = [
        {"provider": "openai", "model": "gpt-4.1-mini"},
        {"provider": "openai", "model": "gpt-4.1"},
        {"provider": "lm-studio", "model": "unsloth/qwen3-30b-a3b"},
        # Add more models as needed
    ]
    
    concurrent_requests = 10  # Adjust based on API limits
    
    try:
        # Initialize evaluator with dataset
        evaluator = PerformanceEvaluator(
            dataset_path=dataset_path,
            sample_size=5,
            random_state=42,
            content_columns=['content'] if dataset_path else None
        )
        
        print(f"Testing {len(model_configs)} models with {concurrent_requests} concurrent requests each")
        print(f"Sample size: {evaluator.sample_size} prompts per model")
        
        # Prepare prompts once for all models
        if dataset_path:
            prompts = evaluator._prepare_prompts_from_dataset()
        else:
            prompts = evaluator._create_synthetic_prompts()
        
        print(f"Using {'dataset' if dataset_path else 'synthetic'} prompts")
        
        all_results = {}
        total_start_time = asyncio.get_event_loop().time()
        
        # Test each model asynchronously
        for i, config in enumerate(model_configs, 1):
            provider = config['provider']
            model = config['model']
            model_key = f"{provider}_{model}"
            
            print(f"\n[{i}/{len(model_configs)}] Testing {provider} - {model}")
            
            try:
                # Run async performance evaluation
                results = await evaluator.evaluate_model_performance_async(
                    provider=provider,
                    model=model,
                    prompts=prompts,
                    concurrent_requests=concurrent_requests
                )
                
                all_results[model_key] = results
                evaluator.results.extend(results)
                
                # Print individual model summary
                successful_results = [r for r in results if r.success]
                if successful_results:
                    avg_tokens_per_sec = sum(r.tokens_per_second for r in successful_results) / len(successful_results)
                    avg_response_time = sum(r.response_time for r in successful_results) / len(successful_results)
                    success_rate = len(successful_results) / len(results)
                    
                    print(f"Success rate: {success_rate:.1%}")
                    print(f"Avg tokens/sec: {avg_tokens_per_sec:.1f}")
                    print(f"Avg response time: {avg_response_time:.2f}s")
                else:
                    print(f"All requests failed for {model_key}")
                    
            except Exception as e:
                print(f"Failed to test {model_key}: {e}")
                # Add empty results to maintain structure
                all_results[model_key] = []
        
        total_end_time = asyncio.get_event_loop().time()
        total_time = total_end_time - total_start_time
        
        # Print comprehensive results
        print(f"\n" + "="*80)
        print("ASYNC PERFORMANCE EVALUATION RESULTS")
        print(f"Total evaluation time: {total_time:.2f} seconds")
        print("="*80)
        
        if any(all_results.values()):
            evaluator.print_performance_report(all_results)
            
            # Save results
            save_paths = evaluator.save_results(all_results, "performance_results")
            print(f"\nResults saved to: {save_paths['output_directory']}")
            
            # Calculate time savings
            estimated_sequential_time = len(model_configs) * evaluator.sample_size * 2  # Rough estimate
            time_saved = estimated_sequential_time - total_time
            speedup = estimated_sequential_time / total_time if total_time > 0 else 1
            
            print(f"\nPerformance Improvement:")
            print(f"   Estimated sequential time: {estimated_sequential_time:.0f}s")
            print(f"   Actual async time: {total_time:.2f}s") 
            print(f"   Estimated speedup: {speedup:.1f}x")
            print(f"   Time saved: {time_saved:.0f}s ({time_saved/60:.1f} minutes)")
        else:
            print("No successful results to report")
            
        return total_time, all_results
        
    except Exception as e:
        print(f"Async performance evaluation failed: {e}")
        print("Try reducing concurrent_requests or check your API configuration")
        return None, {}

def performance_evaluation_sync():
    """
    Synchronous performance evaluation (fallback)
    """
    print("\n" + "="*80)
    print("SYNC PERFORMANCE EVALUATION")
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
        print("No dataset found, using synthetic prompts for testing.")
    
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
            content_columns=['content'] if dataset_path else None
        )
        
        print(f"Testing {len(model_configs)} models sequentially")
        print(f"Sample size: {evaluator.sample_size} prompts per model")
        
        # Measure sync execution time
        start_time = time.time()
        
        # Run synchronous evaluation
        results = evaluator.evaluate_multiple_models(model_configs)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print results
        print(f"\nTotal sync evaluation time: {total_time:.2f} seconds")
        evaluator.print_performance_report(results)
        
        # Save results
        save_paths = evaluator.save_results(results, "performance_results")
        print(f"\nResults saved to: {save_paths['output_directory']}")
        
        return total_time, results
        
    except Exception as e:
        print(f"Synchronous performance evaluation failed: {e}")
        return None, {}

async def compare_async_vs_sync():
    """
    Compare async vs sync performance evaluation times
    """
    print("="*80)
    print("ASYNC VS SYNC PERFORMANCE COMPARISON")
    print("="*80)
    
    # Run async version
    print("\n>>> RUNNING ASYNC VERSION...")
    async_time, async_results = await performance_evaluation_async()
    
    # Clear evaluator state and run sync version
    print("\n>>> RUNNING SYNC VERSION...")
    sync_time, sync_results = performance_evaluation_sync()
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    if async_time is not None and sync_time is not None:
        speedup = sync_time / async_time if async_time > 0 else 1
        time_saved = sync_time - async_time
        efficiency = (time_saved / sync_time) * 100 if sync_time > 0 else 0
        
        print(f"Async processing time:    {async_time:.2f} seconds")
        print(f"Sync processing time:     {sync_time:.2f} seconds")
        print(f"Time saved:               {time_saved:.2f} seconds")
        print(f"Speedup factor:           {speedup:.2f}x")
        print(f"Efficiency improvement:   {efficiency:.1f}%")
        
        if speedup > 1:
            print(f"\nAsync is {speedup:.2f}x FASTER than sync!")
            print(f"You saved {time_saved:.2f} seconds ({time_saved/60:.2f} minutes)")
        elif speedup < 1:
            print(f"\nSync was {1/speedup:.2f}x faster (possibly due to overhead)")
        else:
            print(f"\nBoth methods took similar time")
            
        # Detailed breakdown
        print(f"\nDetailed Breakdown:")
        print(f"- Async method processed requests concurrently")
        print(f"- Sync method processed requests sequentially") 
        print(f"- Time difference: {abs(time_saved):.2f} seconds")
        
        if async_results and sync_results:
            print(f"- Both methods tested the same models with same parameters")
            print(f"- Results should be comparable (slight variance expected)")
    else:
        print("Could not compare - one or both methods failed")
        if async_time is None:
            print("- Async method failed")
        if sync_time is None:
            print("- Sync method failed")

async def main_async():
    """Run async performance evaluation"""
    print("PERFORMANCE EVALUATION PIPELINE - ASYNC VERSION")
    print("Using concurrent requests for faster testing")
    await performance_evaluation_async()

def main():
    """Main function with async/sync options"""
    import sys
    
    print("PERFORMANCE EVALUATION PIPELINE")
    print()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--sync":
            print("Running synchronous version...")
            performance_evaluation_sync()
        elif sys.argv[1] == "--compare":
            print("Running comparison between async and sync...")
            try:
                asyncio.run(compare_async_vs_sync())
            except KeyboardInterrupt:
                print("\nProcess interrupted by user")
            except Exception as e:
                print(f"\nComparison failed: {e}")
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python performance_eval.py           # Run async version (default)")
            print("  python performance_eval.py --sync    # Run sync version only")
            print("  python performance_eval.py --compare # Compare async vs sync")
            print("  python performance_eval.py --help    # Show this help")
            return
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            return
    else:
        # Run async version by default
        try:
            asyncio.run(main_async())
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
        except Exception as e:
            print(f"\nAsync processing failed: {e}")
            print("Attempting fallback to synchronous processing...")
            performance_evaluation_sync()
    
    print("\n" + "="*80)
    print("PERFORMANCE EVALUATION COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main() 