#!/usr/bin/env python3
"""
Example usage of the LLM Annotation Pipeline with Async Support

This script demonstrates how to use the annotation pipeline to generate explanations
for why content is classified as scam or legitimate using concurrent requests.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.annotation_pipeline import LLMAnnotationPipeline

async def annotation_async():
    print("="*80)
    print("ASYNC ANNOTATION: Generating explanations for scam/legitimate classifications")
    print("="*80)
    
    dataset_path = "data/cleaned/unified_phishing_email_dataset.csv"
    provider = "lm-studio"
    model = "unsloth/qwen3-30b-a3b"
    sample_size = 10
    concurrent_requests = 5  # Adjust based on your API limits and system capacity
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please ensure the dataset file exists in the current directory.")
        return
    
    try:
        # Initialize annotation pipeline
        pipeline = LLMAnnotationPipeline(
            dataset_path=dataset_path,
            provider=provider,
            model=model,
            sample_size=sample_size,
            random_state=42,
            content_columns=['subject', 'body'],
            balanced_sample=False,
            output_dir="results/annotated",
            enable_thinking=False,
            use_structure_model=False
        )
        
        print(f"Starting async annotation with {concurrent_requests} concurrent requests...")
        print(f"Processing {sample_size} samples...")
        
        # Run async annotation
        start_time = asyncio.get_event_loop().time()
        results = await pipeline.run_full_annotation_async(concurrent_requests=concurrent_requests)
        end_time = asyncio.get_event_loop().time()
        
        total_time = end_time - start_time
        
        print("\nAsync annotation completed successfully!")
        print(f"Results saved to: {results['save_paths']['results_directory']}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per annotation: {total_time/sample_size:.3f} seconds")
        print(f"Estimated speedup vs sequential: ~{concurrent_requests}x")
        
        # Print summary statistics
        summary = results['summary']
        print(f"\nSummary:")
        print(f"   Total annotations: {summary['total_records']}")
        print(f"   Successful annotations: {summary['successful_annotations']}")
        print(f"   Success rate: {summary['successful_annotations']/summary['total_records']:.2%}")
        
    except Exception as e:
        print(f"Async annotation failed: {e}")
        print("Try reducing concurrent_requests or check your API configuration")

def annotation_sync():
    """Fallback synchronous annotation method"""
    print("="*80)
    print("SYNC ANNOTATION: Generating explanations for scam/legitimate classifications")
    print("="*80)
    
    dataset_path = "data/cleaned/unified_phishing_email_dataset.csv"
    provider = "lm-studio"
    model = "unsloth/qwen3-30b-a3b"
    sample_size = 217204
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please ensure the dataset file exists in the current directory.")
        return
    
    try:
        # Initialize annotation pipeline
        pipeline = LLMAnnotationPipeline(
            dataset_path=dataset_path,
            provider=provider,
            model=model,
            sample_size=sample_size,
            random_state=42,
            content_columns=['subject', 'body'],
            balanced_sample=False,
            output_dir="results/annotated",
            enable_thinking=False,
            use_structure_model=False
        )
        
        # Run synchronous annotation
        results = pipeline.run_full_annotation()
        
        print("\n✅ Synchronous annotation completed successfully!")
        print(f"📁 Results saved to: {results['save_paths']['results_directory']}")
        
    except Exception as e:
        print(f"❌ Synchronous annotation failed: {e}")

async def main_async():
    """Run async annotation example"""
    print("SCAM DETECTION ANNOTATION PIPELINE - ASYNC VERSION")
    print("Using concurrent requests for faster processing")
    await annotation_async()

def main():
    """Main function with async/sync options"""
    import sys
    
    # Check if user wants to run sync version
    if len(sys.argv) > 1 and sys.argv[1] == "--sync":
        print("SCAM DETECTION ANNOTATION PIPELINE - SYNC VERSION")
        annotation_sync()
    else:
        # Run async version by default
        try:
            asyncio.run(main_async())
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
        except Exception as e:
            print(f"\nAsync processing failed: {e}")
            print("Attempting fallback to synchronous processing...")
            annotation_sync()

if __name__ == "__main__":
    main() 