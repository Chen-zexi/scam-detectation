#!/usr/bin/env python3
"""
Example: Transcript Generation for Scam Detection

This script demonstrates how to use the transcript generation pipeline
to create realistic phone conversation datasets for scam detection training.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.transcript_generator import TranscriptGenerator
from src.transcript_prompts import MODEL_A_CONFIG, MODEL_B_CONFIG

async def example_transcript_generation():
    """Example of generating transcripts with custom configuration"""
    print("="*80)
    print("TRANSCRIPT GENERATION EXAMPLE")
    print("="*80)
    
    # Custom model configurations
    custom_model_a = {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "categories": {
            "authority_scam": {"percentage": 40},
            "tech_scam": {"percentage": 30},
            "legitimate_half_a": {"percentage": 25},
            "subtle_scam": {"percentage": 5}
        }
    }
    
    custom_model_b = {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "categories": {
            "urgency_scam": {"percentage": 30},
            "financial_scam": {"percentage": 25},
            "legitimate_half_b": {"percentage": 30},
            "borderline": {"percentage": 10},
            "subtle_scam": {"percentage": 5}
        }
    }
    
    try:
        # Initialize generator with custom configuration
        generator = TranscriptGenerator(
            sample_size=50,  # Small sample for demonstration
            model_a_config=custom_model_a,
            model_b_config=custom_model_b,
            output_dir="example_results",
            enable_thinking=True,
            use_structure_model=False
        )
        
        print("Configuration:")
        print(f"  Sample size: {generator.sample_size}")
        print(f"  Model A: {custom_model_a['provider']} - {custom_model_a['model']}")
        print(f"  Model B: {custom_model_b['provider']} - {custom_model_b['model']}")
        print(f"  Output directory: {generator.output_dir}")
        
        # Calculate distribution
        distribution = generator.calculate_category_distribution()
        print(f"\nCategory distribution:")
        for category, count in distribution.items():
            print(f"  {category}: {count}")
        
        print(f"\nGenerating transcripts...")
        
        # Generate transcripts
        results = await generator.generate_transcripts_async(concurrent_requests=3)
        
        # Save results
        save_results = generator.save_transcripts(results)
        
        print(f"\nGeneration completed!")
        print(f"  Successful: {save_results['success_count']}")
        print(f"  Errors: {save_results['error_count']}")
        print(f"  Success rate: {save_results['success_count'] / len(results):.2%}")
        print(f"  Results saved to: {save_results['detailed_results']}")
        
        # Display sample results
        if save_results['success_count'] > 0:
            print(f"\nSample results:")
            import pandas as pd
            df = pd.read_csv(save_results['detailed_results'])
            
            # Show distribution of classifications
            print(f"Classification distribution:")
            for classification, count in df['classification'].value_counts().items():
                print(f"  {classification}: {count}")
            
            # Show sample transcript
            sample = df[df['classification'] == 'OBVIOUS_SCAM'].iloc[0] if len(df[df['classification'] == 'OBVIOUS_SCAM']) > 0 else df.iloc[0]
            print(f"\nSample transcript ({sample['classification']}):")
            print(f"Category: {sample['category_assigned']}")
            print(f"Length: {sample['conversation_length']} words")
            print(f"Demographics: {sample['participant_demographics']}")
            print(f"Transcript preview: {sample['transcript'][:300]}...")
        
        return True
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main example function"""
    success = await example_transcript_generation()
    
    if success:
        print("\n✅ Transcript generation example completed successfully!")
        print("\nNext steps:")
        print("1. Run the full pipeline: uv run python main.py")
        print("2. Evaluate generated transcripts: uv run python transcript_eval.py")
        print("3. Use generated data for model training")
    else:
        print("\n❌ Transcript generation example failed")
        print("Please check the error messages above")

if __name__ == "__main__":
    asyncio.run(main()) 