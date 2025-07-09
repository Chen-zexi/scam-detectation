#!/usr/bin/env python3
"""
Example usage of the Transcript Generation Evaluation Pipeline

This script demonstrates how to evaluate generated transcript datasets for scam detection.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import asyncio
from src.llm_provider import LLM  # or your LLM wrapper
from datetime import datetime

# Add src to path
sys.path.append('src')

from src import ScamDetectionEvaluator

def transcript_eval():
    print("="*80)
    print("TRANSCRIPT EVALUATION")
    print("="*80)
    
    # Look for generated transcript datasets
    results_dir = Path("results")
    transcript_datasets = []
    
    if results_dir.exists():
        for dataset_dir in results_dir.glob("generated_transcripts/*"):
            detailed_results = dataset_dir / "detailed_results.csv"
            if detailed_results.exists():
                transcript_datasets.append({
                    'path': str(detailed_results),
                    'name': dataset_dir.name,
                    'timestamp': dataset_dir.name
                })
    
    if not transcript_datasets:
        print("No generated transcript datasets found!")
        print("Please run transcript generation first using:")
        print("uv run python src/transcript_generator.py --sample-size 100")
        return
    
    # Use the most recent dataset
    latest_dataset = max(transcript_datasets, key=lambda x: x['timestamp'])
    dataset_path = latest_dataset['path']
    
    print(f"Using dataset: {latest_dataset['name']}")
    print(f"Path: {dataset_path}")
    
    # Load generated transcripts
    df = pd.read_csv(dataset_path)

    # Set up OpenAI o3 model
    llm = LLM(provider="openai", model="o3").get_llm()

    # Define a structured evaluation prompt
    def build_eval_prompt(transcript):
        return f"""
You are an expert scam call evaluator. Read the following phone transcript and classify it as one of:
- OBVIOUS_SCAM
- SUBTLE_SCAM
- BORDERLINE_SUSPICIOUS
- LEGITIMATE

Provide a brief justification for your classification.

Transcript:
{transcript}

Respond in this format:
Classification: <one of the above>
Reason: <your reasoning>
"""

    async def evaluate_transcript(row):
        prompt = build_eval_prompt(row['transcript'])
        # If using LangChain or similar, format as messages if needed
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content if hasattr(response, "content") else str(response)

    async def main():
        results = []
        for idx, row in df.iterrows():
            result = await evaluate_transcript(row)
            results.append(result)
            print(f"Evaluated {idx+1}/{len(df)}")
        df['evaluation'] = results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"results/generated_transcripts/{timestamp}/evaluation_results.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved evaluation results to {out_path}")

    try:
        # Run evaluation
        await asyncio.run(main())
        
        print("\nTranscript Evaluation completed successfully!")
        print(f"Results saved to: {out_path}")
        
    except Exception as e:
        print(f"Transcript Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run transcript evaluation"""
    print("SCAM DETECTION PIPELINE - TRANSCRIPT EVALUATION")
    transcript_eval()

if __name__ == "__main__":
    main() 