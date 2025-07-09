#!/usr/bin/env python3
"""
Transcript Generation Pipeline for Scam Detection

This pipeline generates realistic phone conversation transcripts for scam detection training.
It uses alternating models to create diverse, realistic conversations that can be used to
train and evaluate scam detection models.

Usage:
    python transcript_generator.py --sample-size 1000 --checkpoint-interval 100
"""

import argparse
import sys
import os
import asyncio
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add current directory to Python path to allow imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from api_provider import LLM
from api_call import make_api_call, parse_structured_output, make_api_call_async, parse_structured_output_async
from results_saver import ResultsSaver
from transcript_prompts import get_model_config, get_prompt_for_category, MODEL_A_CONFIG, MODEL_B_CONFIG
from pydantic import BaseModel

class TranscriptResponseSchema(BaseModel):
    transcript: str
    classification: str  # "LEGITIMATE", "OBVIOUS_SCAM", "BORDERLINE_SUSPICIOUS", "SUBTLE_SCAM"
    category_assigned: str
    conversation_length: int
    participant_demographics: str
    timestamp: str

class TranscriptGenerator:
    """Generate realistic phone conversation transcripts for scam detection training"""
    
    def __init__(self, 
                 sample_size: int = 1000,
                 model_a_config: Dict = None,
                 model_b_config: Dict = None,
                 output_dir: str = "results",
                 enable_thinking: bool = False,
                 use_structure_model: bool = False,
                 selected_model: str = None,
                 selected_provider: str = None):
        """
        Initialize the transcript generator
        
        Args:
            sample_size: Total number of transcripts to generate
            model_a_config: Configuration for Model A (authority scams, tech support, legitimate customer service)
            model_b_config: Configuration for Model B (urgency scams, financial fraud, legitimate personal/business, borderline)
            output_dir: Directory to save generated transcripts
            enable_thinking: Whether to enable thinking tokens in prompts
            use_structure_model: Whether to use structured output parsing
            selected_model: The model selected from the main pipeline (e.g., 'gpt-4.1-mini')
            selected_provider: The provider selected from the main pipeline (e.g., 'openai')
        """
        self.sample_size = sample_size
        self.model_a_config = model_a_config or MODEL_A_CONFIG
        self.model_b_config = model_b_config or MODEL_B_CONFIG
        self.output_dir = output_dir
        self.enable_thinking = enable_thinking
        self.use_structure_model = use_structure_model
        self.selected_model = selected_model
        self.selected_provider = selected_provider
        
        # Initialize components
        self.model_a_llm = None
        self.model_b_llm = None
        self.structure_model = None
        self.transcripts = []
        
        # Checkpoint state
        self.current_index = 0
        self.checkpoint_file = None
        self.start_time = None
        self.last_checkpoint_message = None
        
    def setup_models(self):
        """Initialize the LLM models"""
        try:
            # Debug: Print what we received
            print(f"DEBUG: selected_model = {self.selected_model}")
            print(f"DEBUG: selected_provider = {self.selected_provider}")
            
            # Use the selected model from the main pipeline if provided
            if self.selected_model and self.selected_provider:
                # Both models will use the same selected model but with different configurations
                self.model_a_llm = LLM(provider=self.selected_provider, model=self.selected_model).get_llm()
                self.model_b_llm = LLM(provider=self.selected_provider, model=self.selected_model).get_llm()
                
                print(f"Model A initialized: {self.selected_provider} - {self.selected_model}")
                print(f"Model B initialized: {self.selected_provider} - {self.selected_model}")
            else:
                # Fall back to default configurations
                model_a_provider = self.model_a_config["provider"]
                model_a_model = self.model_a_config["model"]
                self.model_a_llm = LLM(provider=model_a_provider, model=model_a_model).get_llm()
                
                model_b_provider = self.model_b_config["provider"]
                model_b_model = self.model_b_config["model"]
                self.model_b_llm = LLM(provider=model_b_provider, model=model_b_model).get_llm()
                
                print(f"Model A initialized: {model_a_provider} - {model_a_model}")
                print(f"Model B initialized: {model_b_provider} - {model_b_model}")
            
            # Setup structure model if needed
            if self.use_structure_model:
                self.structure_model = LLM(provider='lm-studio', model='osmosis-structure-0.6b@f16').get_llm()
            
        except Exception as e:
            raise Exception(f"Error initializing models: {e}")
    
    def get_base_prompt(self) -> str:
        """Generate base system prompt for transcript generation"""
        return """
You are an expert in generating realistic phone conversation transcripts for scam detection training.
Your task is to create authentic, detailed phone conversations that can be used to train AI models
to detect various types of scams and legitimate calls.

GUIDELINES:
- Generate realistic, natural-sounding phone conversations
- Include proper speaker identification: [CALLER] and [RECIPIENT]
- Vary conversation length between 2-4 minutes of dialogue
- Include realistic details like background sounds, interruptions, natural speech patterns
- Make conversations feel authentic and believable
- Include appropriate demographics and context for participants

OUTPUT FORMAT:
- transcript: The complete phone conversation with speaker identification
- classification: LEGITIMATE, OBVIOUS_SCAM, BORDERLINE_SUSPICIOUS, or SUBTLE_SCAM
- category_assigned: The specific category of the conversation
- conversation_length: Approximate length in words
- participant_demographics: Brief description of participants (e.g., "adult_female_customer_service", "elderly_male_victim")
- timestamp: Current timestamp in ISO format

CONVERSATION ELEMENTS TO INCLUDE:
- Natural speech patterns and interruptions
- Realistic background sounds: [typing], [office noise], [phone ringing]
- Appropriate emotional responses and reactions
- Realistic timing and pacing
- Authentic dialogue that matches the scenario
"""

    def create_generation_prompt(self, category: str, model_type: str) -> str:
        """Create generation prompt for a specific category"""
        base_prompt = self.get_base_prompt()
        category_prompt = get_prompt_for_category(category)
        
        # Add model-specific instructions
        model_instructions = f"""
MODEL TYPE: {model_type}
CATEGORY: {category}

Generate a realistic phone conversation transcript following the category-specific guidelines above.
Ensure the conversation is authentic, detailed, and appropriate for scam detection training.

Please provide your response in the following format:
- A complete phone conversation transcript with speaker identification
- The classification (LEGITIMATE, OBVIOUS_SCAM, BORDERLINE_SUSPICIOUS, or SUBTLE_SCAM)
- The category assigned
- Approximate conversation length in words
- Brief participant demographics
- Current timestamp
"""
        
        full_prompt = f"{base_prompt}\n\n{category_prompt}\n\n{model_instructions}"
        return full_prompt

    def calculate_category_distribution(self) -> Dict[str, int]:
        """Calculate how many transcripts to generate for each category"""
        distribution = {}
        
        # Model A categories
        for category, config in self.model_a_config["categories"].items():
            count = int(self.sample_size * config["percentage"] / 100)
            distribution[category] = count
        
        # Model B categories
        for category, config in self.model_b_config["categories"].items():
            count = int(self.sample_size * config["percentage"] / 100)
            distribution[category] = count
        
        # Adjust for rounding errors
        total_allocated = sum(distribution.values())
        if total_allocated != self.sample_size:
            # Add remaining to largest category
            largest_category = max(distribution, key=distribution.get)
            distribution[largest_category] += (self.sample_size - total_allocated)
        
        return distribution

    def _create_transcript_record(self, 
                                 record_id: int, 
                                 response: TranscriptResponseSchema,
                                 category: str,
                                 model_type: str) -> Dict[str, Any]:
        """Create a transcript record for saving"""
        return {
            'id': record_id,
            'transcript': response.transcript,
            'classification': response.classification,
            'generation_model': f"model_{model_type.lower()}",
            'category_assigned': response.category_assigned,
            'conversation_length': response.conversation_length,
            'participant_demographics': response.participant_demographics,
            'timestamp': response.timestamp,
            'model_type': model_type,
            'category': category
        }

    def _create_error_transcript_record(self, record_id: int, category: str, model_type: str, error_message: str) -> Dict[str, Any]:
        """Create an error record for failed generations"""
        return {
            'id': record_id,
            'transcript': f"ERROR: {error_message}",
            'classification': 'ERROR',
            'generation_model': f"model_{model_type.lower()}",
            'category_assigned': category,
            'conversation_length': 0,
            'participant_demographics': 'error',
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'category': category,
            'error': error_message
        }

    async def generate_single_transcript(self, category: str, model_type: str) -> Dict[str, Any]:
        """Generate a single transcript for a specific category"""
        try:
            # Select appropriate model
            llm = self.model_a_llm if model_type == "A" else self.model_b_llm
            
            # Create prompt
            prompt = self.create_generation_prompt(category, model_type)
            
            # Generate transcript using direct LLM call
            from langchain_core.prompts import ChatPromptTemplate
            
            # Create a simple prompt template
            prompt_template = ChatPromptTemplate.from_template(prompt)
            messages = prompt_template.format_messages()
            
            # Make the API call directly
            if hasattr(llm, 'ainvoke'):
                response = await llm.ainvoke(messages)
            else:
                response = llm.invoke(messages)
            
            # Extract the content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Create a basic response structure
            # In a real implementation, you'd want to parse the response more carefully
            response_obj = TranscriptResponseSchema(
                transcript=response_text[:1000] + "..." if len(response_text) > 1000 else response_text,
                classification='OBVIOUS_SCAM' if any(word in response_text.lower() for word in ['scam', 'fraud', 'fake']) else 'LEGITIMATE',
                category_assigned=category,
                conversation_length=len(response_text.split()),
                participant_demographics='adult_mixed',
                timestamp=datetime.now().isoformat()
            )
            
            return {
                'success': True,
                'response': response_obj,
                'category': category,
                'model_type': model_type
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'category': category,
                'model_type': model_type
            }

    async def generate_transcripts_async(self, concurrent_requests: int = 5) -> List[Dict[str, Any]]:
        """Generate transcripts asynchronously with the specified distribution"""
        print(f"\nGenerating {self.sample_size} transcripts...")
        
        # Calculate distribution
        distribution = self.calculate_category_distribution()
        print(f"Category distribution: {distribution}")
        
        # Create generation tasks
        tasks = []
        record_id = 0
        
        for category, count in distribution.items():
            # Determine model type based on category
            if category in self.model_a_config["categories"]:
                model_type = "A"
            elif category in self.model_b_config["categories"]:
                model_type = "B"
            else:
                print(f"Warning: Unknown category {category}, skipping")
                continue
            
            for _ in range(count):
                tasks.append((record_id, category, model_type))
                record_id += 1
        
        # Shuffle tasks for better distribution
        random.shuffle(tasks)
        
        # Process with semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent_requests)
        results = []
        
        async def process_with_semaphore(task):
            record_id, category, model_type = task
            async with semaphore:
                result = await self.generate_single_transcript(category, model_type)
                result['record_id'] = record_id
                return result
        
        # Process tasks with progress bar
        with tqdm(total=len(tasks), desc="Generating transcripts") as pbar:
            for i in range(0, len(tasks), concurrent_requests):
                batch = tasks[i:i + concurrent_requests]
                batch_results = await asyncio.gather(*[process_with_semaphore(task) for task in batch])
                results.extend(batch_results)
                pbar.update(len(batch))
        
        return results

    def save_transcripts(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Save generated transcripts to files"""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = "generated_transcripts"
        results_dir = Path(self.output_dir) / dataset_name / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        successful_results = []
        error_results = []
        
        for result in results:
            if result['success']:
                record = self._create_transcript_record(
                    result['record_id'],
                    result['response'],
                    result['category'],
                    result['model_type']
                )
                successful_results.append(record)
            else:
                record = self._create_error_transcript_record(
                    result['record_id'],
                    result['category'],
                    result['model_type'],
                    result['error']
                )
                error_results.append(record)
        
        # Save detailed results
        detailed_df = pd.DataFrame(successful_results + error_results)
        detailed_path = results_dir / "detailed_results.csv"
        detailed_df.to_csv(detailed_path, index=False)
        
        # Save summary statistics
        summary_stats = {
            'total_generated': len(results),
            'successful': len(successful_results),
            'errors': len(error_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            'category_distribution': detailed_df['category_assigned'].value_counts().to_dict(),
            'classification_distribution': detailed_df['classification'].value_counts().to_dict(),
            'model_distribution': detailed_df['generation_model'].value_counts().to_dict(),
            'average_conversation_length': detailed_df['conversation_length'].mean(),
            'generation_timestamp': timestamp,
            'model_a_config': self.model_a_config,
            'model_b_config': self.model_b_config
        }
        
        # Save summary
        summary_path = results_dir / "generation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Save human-readable summary
        summary_report_path = results_dir / "summary_report.txt"
        with open(summary_report_path, 'w') as f:
            f.write("TRANSCRIPT GENERATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total generated: {len(results)}\n")
            f.write(f"Successful: {len(successful_results)}\n")
            f.write(f"Errors: {len(error_results)}\n")
            f.write(f"Success rate: {summary_stats['success_rate']:.2%}\n\n")
            
            f.write("Category Distribution:\n")
            for category, count in summary_stats['category_distribution'].items():
                f.write(f"  {category}: {count}\n")
            
            f.write("\nClassification Distribution:\n")
            for classification, count in summary_stats['classification_distribution'].items():
                f.write(f"  {classification}: {count}\n")
            
            f.write(f"\nAverage conversation length: {summary_stats['average_conversation_length']:.1f} words\n")
            f.write(f"Generation completed: {timestamp}\n")
        
        return {
            'detailed_results': str(detailed_path),
            'summary_json': str(summary_path),
            'summary_report': str(summary_report_path),
            'success_count': len(successful_results),
            'error_count': len(error_results),
            'total_count': len(results)
        }

    def _get_checkpoint_filename(self) -> str:
        """Generate checkpoint filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"transcript_generation_{timestamp}.json"

    def _find_existing_checkpoint(self, checkpoint_dir: str = "checkpoints") -> Optional[Path]:
        """Find existing checkpoint files"""
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None
        
        checkpoint_files = list(checkpoint_path.glob("transcript_generation_*.json"))
        if not checkpoint_files:
            return None
        
        # Return most recent checkpoint
        return max(checkpoint_files, key=lambda x: x.stat().st_mtime)

    def load_checkpoint(self, checkpoint_dir: str = "checkpoints", 
                       resume_from_checkpoint: bool = True,
                       override_compatibility: bool = False) -> bool:
        """Load checkpoint if available"""
        if not resume_from_checkpoint:
            return False
        
        checkpoint_file = self._find_existing_checkpoint(checkpoint_dir)
        if not checkpoint_file:
            return False
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Validate checkpoint compatibility
            if not override_compatibility:
                if (checkpoint_data.get('model_a_config') != self.model_a_config or
                    checkpoint_data.get('model_b_config') != self.model_b_config):
                    print("Checkpoint configuration doesn't match current configuration.")
                    print("Use --override-compatibility to force resume.")
                    return False
            
            # Load checkpoint data
            self.transcripts = checkpoint_data.get('transcripts', [])
            self.current_index = len(self.transcripts)
            self.checkpoint_file = str(checkpoint_file)
            
            print(f"Resumed from checkpoint: {checkpoint_file}")
            print(f"Already processed: {self.current_index} transcripts")
            
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def save_checkpoint(self, checkpoint_dir: str = "checkpoints"):
        """Save current progress to checkpoint"""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)
        
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'current_index': self.current_index,
            'total_target': self.sample_size,
            'transcripts': self.transcripts,
            'model_a_config': self.model_a_config,
            'model_b_config': self.model_b_config,
            'progress': self.current_index / self.sample_size if self.sample_size > 0 else 0
        }
        
        if not self.checkpoint_file:
            self.checkpoint_file = str(checkpoint_path / self._get_checkpoint_filename())
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self._write_checkpoint_message(f"Checkpoint saved: {self.current_index}/{self.sample_size}")

    def _write_checkpoint_message(self, message: str):
        """Write checkpoint message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    async def process_full_generation_with_checkpoints(self, 
                                                     checkpoint_interval: int = 100,
                                                     checkpoint_dir: str = "checkpoints",
                                                     resume_from_checkpoint: bool = True,
                                                     concurrent_requests: int = 5,
                                                     override_compatibility: bool = False) -> Dict[str, Any]:
        """Generate transcripts with checkpointing support"""
        
        # Try to resume from checkpoint
        if resume_from_checkpoint:
            self.load_checkpoint(checkpoint_dir, resume_from_checkpoint, override_compatibility)
        
        if self.current_index >= self.sample_size:
            print("Generation already completed!")
            return {'status': 'completed', 'transcripts': self.transcripts}
        
        # Setup models
        self.setup_models()
        
        # Calculate remaining work
        remaining_count = self.sample_size - self.current_index
        print(f"Generating {remaining_count} additional transcripts...")
        
        # Calculate distribution for remaining work
        distribution = self.calculate_category_distribution()
        total_allocated = sum(distribution.values())
        
        # Scale distribution for remaining work
        scale_factor = remaining_count / total_allocated if total_allocated > 0 else 1
        remaining_distribution = {k: int(v * scale_factor) for k, v in distribution.items()}
        
        # Create tasks for remaining work
        tasks = []
        for category, count in remaining_distribution.items():
            if category in self.model_a_config["categories"]:
                model_type = "A"
            elif category in self.model_b_config["categories"]:
                model_type = "B"
            else:
                continue
            
            for _ in range(count):
                tasks.append((self.current_index + len(tasks), category, model_type))
        
        # Process with checkpointing
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def process_with_semaphore(task):
            record_id, category, model_type = task
            async with semaphore:
                result = await self.generate_single_transcript(category, model_type)
                result['record_id'] = record_id
                return result
        
        # Process in batches with checkpointing
        batch_size = checkpoint_interval
        all_results = []
        
        with tqdm(total=len(tasks), desc="Generating transcripts") as pbar:
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                
                # Process batch
                batch_results = await asyncio.gather(*[process_with_semaphore(task) for task in batch])
                all_results.extend(batch_results)
                
                # Update progress
                self.current_index += len(batch_results)
                pbar.update(len(batch_results))
                
                # Save checkpoint
                if (i + batch_size) % checkpoint_interval == 0 or i + batch_size >= len(tasks):
                    self.save_checkpoint(checkpoint_dir)
        
        # Save final results
        save_results = self.save_transcripts(all_results)
        
        return {
            'status': 'completed',
            'results': save_results,
            'total_generated': len(all_results),
            'checkpoint_file': self.checkpoint_file
        }

    async def run_full_generation(self) -> Dict[str, Any]:
        """Run the complete transcript generation process"""
        print("="*80)
        print("TRANSCRIPT GENERATION PIPELINE")
        print("="*80)
        
        try:
            # Setup models
            self.setup_models()
            
            # Generate transcripts
            results = await self.generate_transcripts_async()
            
            # Save results
            save_results = self.save_transcripts(results)
            
            print(f"\nGeneration completed successfully!")
            print(f"Generated: {save_results['success_count']} transcripts")
            print(f"Errors: {save_results['error_count']} transcripts")
            print(f"Results saved to: {save_results['detailed_results']}")
            
            return {
                'status': 'success',
                'results': save_results,
                'total_generated': len(results)
            }
            
        except Exception as e:
            print(f"Generation failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate transcript dataset for scam detection")
    
    parser.add_argument("--sample-size", type=int, default=1000,
                       help="Number of transcripts to generate")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                       help="Save checkpoint every N transcripts")
    parser.add_argument("--concurrent-requests", type=int, default=5,
                       help="Number of concurrent API requests")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory for checkpoint files")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing checkpoint")
    parser.add_argument("--override-compatibility", action="store_true",
                       help="Override checkpoint compatibility check")
    parser.add_argument("--enable-thinking", action="store_true",
                       help="Enable thinking tokens in prompts")
    parser.add_argument("--use-structure-model", action="store_true",
                       help="Use structured output parsing")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    
    return parser.parse_args()

async def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize generator
    generator = TranscriptGenerator(
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        enable_thinking=args.enable_thinking,
        use_structure_model=args.use_structure_model
    )
    
    # Run generation with checkpointing
    results = await generator.process_full_generation_with_checkpoints(
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume,
        concurrent_requests=args.concurrent_requests,
        override_compatibility=args.override_compatibility
    )
    
    print(f"\nGeneration completed with status: {results['status']}")
    if 'results' in results:
        print(f"Results saved to: {results['results']['detailed_results']}")

if __name__ == "__main__":
    asyncio.run(main()) 