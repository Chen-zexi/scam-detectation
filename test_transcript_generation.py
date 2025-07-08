#!/usr/bin/env python3
"""
Test script for transcript generation functionality

This script tests the transcript generation pipeline with a small sample size
to verify everything works correctly.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.transcript_generator import TranscriptGenerator

async def test_transcript_generation():
    """Test transcript generation with a small sample"""
    print("="*80)
    print("TESTING TRANSCRIPT GENERATION")
    print("="*80)
    
    try:
        # Initialize generator with small sample size
        generator = TranscriptGenerator(
            sample_size=10,  # Small sample for testing
            output_dir="test_results",
            enable_thinking=False,
            use_structure_model=False,
            selected_model="gpt-4.1-mini",  # Use newer model
            selected_provider="openai"
        )
        
        print("Initializing models...")
        generator.setup_models()
        
        print("Generating transcripts...")
        results = await generator.generate_transcripts_async(concurrent_requests=2)
        
        print("Saving results...")
        save_results = generator.save_transcripts(results)
        
        print(f"\nTest completed successfully!")
        print(f"Generated: {save_results['success_count']} transcripts")
        print(f"Errors: {save_results['error_count']} transcripts")
        print(f"Results saved to: {save_results['detailed_results']}")
        
        # Display sample of generated transcripts
        if save_results['success_count'] > 0:
            print(f"\nSample generated transcript:")
            import pandas as pd
            df = pd.read_csv(save_results['detailed_results'])
            if len(df) > 0:
                sample = df.iloc[0]
                print(f"Classification: {sample['classification']}")
                print(f"Category: {sample['category_assigned']}")
                print(f"Length: {sample['conversation_length']} words")
                print(f"Demographics: {sample['participant_demographics']}")
                print(f"Transcript preview: {sample['transcript'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    success = await test_transcript_generation()
    
    if success:
        print("\n✅ Transcript generation test PASSED")
        print("You can now use the full pipeline with:")
        print("  uv run python main.py")
        print("  # Choose option 3 for transcript generation")
    else:
        print("\n❌ Transcript generation test FAILED")
        print("Please check the error messages above")

if __name__ == "__main__":
    asyncio.run(main()) 