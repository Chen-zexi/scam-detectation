#!/usr/bin/env python3
"""
Generic Synthesis Generator

This module provides a generic generator that can synthesize various types of
scam detection training data based on JSON configuration files.
"""

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Type, Tuple
from datetime import datetime
from enum import Enum
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel

from src.llm_core.api_provider import LLM
from src.llm_core.api_call import make_api_call_async
from src.synthesize.schema_builder import SchemaBuilder
from src.synthesize.synthesis_prompts import SynthesisPromptsManager
from src.synthesize.diversity_manager import DiversityManager, DiversityConfig, DiversityLevel
from src.database import get_scam_data_service
from src.exceptions import UnknownSynthesisTypeError, ModelInitializationError, APICallError

logger = logging.getLogger(__name__)


class SynthesisGenerator:
    """
    Generic generator for synthesizing various types of scam detection training data.
    Uses JSON configuration to define synthesis types, schemas, and prompts.
    """
    
    def __init__(self,
                 synthesis_type: str,
                 sample_size: int = 100,
                 output_dir: str = "results/synthesis",
                 config_path: str = None,
                 provider: str = "openai",
                 model: str = "gpt-4o-mini",
                 enable_thinking: bool = False,
                 use_structure_model: bool = False,
                 save_to_mongodb: bool = True,
                 category: str = "ALL",
                 enable_diversity: bool = False,
                 diversity_level: str = "medium",
                 diversity_config_path: str = None):
        """
        Initialize the synthesis generator.
        
        Args:
            synthesis_type: Type of data to synthesize (e.g., 'phone_transcript', 'phishing_email')
            sample_size: Number of items to generate
            output_dir: Directory to save generated data
            config_path: Path to synthesis configuration JSON
            provider: LLM provider to use
            model: Model name
            enable_thinking: Whether to enable thinking tokens
            use_structure_model: Whether to use structured output parsing
            save_to_mongodb: Whether to save results to MongoDB database
            category: Specific category to generate or 'ALL' for mixed dataset
            enable_diversity: Whether to enable diversity enhancement techniques
            diversity_level: Level of diversity enhancement ('minimal', 'medium', 'high', 'maximum')
            diversity_config_path: Path to diversity configuration JSON
        """
        self.synthesis_type = synthesis_type
        self.sample_size = sample_size
        self.output_dir = Path(output_dir)
        self.provider = provider
        self.model = model
        self.enable_thinking = enable_thinking
        self.use_structure_model = use_structure_model
        self.save_to_mongodb = save_to_mongodb
        self.category = category
        
        # Diversity enhancement settings
        self.enable_diversity = enable_diversity
        self.diversity_level = diversity_level
        self.diversity_config_path = diversity_config_path
        
        # Initialize components
        self.prompts_manager = SynthesisPromptsManager(config_path)
        self.schema_builder = SchemaBuilder()
        
        # Initialize diversity manager if enabled
        self.diversity_manager = None
        if self.enable_diversity:
            try:
                self.diversity_manager = DiversityManager(
                    self.prompts_manager, 
                    self.diversity_config_path
                )
                logger.info(f"Diversity enhancement enabled at {diversity_level} level")
            except Exception as e:
                logger.warning(f"Failed to initialize diversity manager: {e}. Falling back to standard generation.")
                self.enable_diversity = False
        
        # Validate synthesis type
        if synthesis_type not in self.prompts_manager.get_synthesis_types():
            raise UnknownSynthesisTypeError(f"Unknown synthesis type: {synthesis_type}. Available types: {', '.join(self.prompts_manager.get_synthesis_types())}")
        
        # Build response schema - use only LLM fields for the model
        llm_schema_def = self.prompts_manager.get_llm_response_schema(synthesis_type)
        self.response_model = self.schema_builder.build_response_schema(
            f"{synthesis_type.title()}Response",
            llm_schema_def
        )
        
        # Initialize LLM
        self.llm = None
        self.structure_model = None
        
        # Checkpoint state
        self.generated_items = []
        self.current_index = 0
        self.checkpoint_file = None
        self.start_time = None
    
    def setup_models(self):
        """Initialize the LLM models."""
        try:
            self.llm = LLM(provider=self.provider, model=self.model).get_llm()
            logger.info(f"Model initialized: {self.provider} - {self.model}")
            
            if self.use_structure_model:
                llm_provider = LLM()
                self.structure_model = llm_provider.get_structure_model()
                logger.info("Structure model initialized for parsing")
                
        except Exception as e:
            raise ModelInitializationError(f"Error initializing models: {e}") from e
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the current synthesis type."""
        base_prompt = self.prompts_manager.get_system_prompt(self.synthesis_type)
        
        # Add response format instructions using LLM schema only
        schema_info = self.prompts_manager.get_llm_response_schema(self.synthesis_type)
        field_descriptions = []
        for field_name, field_def in schema_info.items():
            desc = field_def.get("description", field_name)
            field_type = field_def.get("type", "string")
            field_descriptions.append(f"- {field_name}: {desc} ({field_type})")
        
        format_instructions = f"""
        
OUTPUT FORMAT:
Please provide your response with the following fields:
{chr(10).join(field_descriptions)}

Ensure all fields are included in your response."""
        
        return base_prompt + format_instructions
    
    def create_generation_prompt(self, category: str) -> str:
        """Create a generation prompt for a specific category."""
        if self.enable_diversity and self.diversity_manager:
            # Create diversity configuration based on level
            diversity_config = self._create_diversity_config()
            return self.diversity_manager.create_diverse_generation_prompt(
                self.synthesis_type, category, diversity_config
            )
        else:
            # Fall back to standard prompt generation
            return self.prompts_manager.create_generation_prompt(self.synthesis_type, category)
    
    def _create_diversity_config(self) -> DiversityConfig:
        """Create diversity configuration based on the selected level."""
        level_map = {
            "minimal": DiversityLevel.MINIMAL,
            "medium": DiversityLevel.MEDIUM,
            "high": DiversityLevel.HIGH,
            "maximum": DiversityLevel.MAXIMUM
        }
        
        level = level_map.get(self.diversity_level.lower(), DiversityLevel.MEDIUM)
        
        # Configure techniques based on level
        if level == DiversityLevel.MINIMAL:
            return DiversityConfig(
                level=level,
                template_variation=True,
                context_injection=False,
                few_shot_learning=False,
                persona_variation=False,
                self_consistency=False
            )
        elif level == DiversityLevel.MEDIUM:
            return DiversityConfig(
                level=level,
                template_variation=True,
                context_injection=True,
                few_shot_learning=True,
                persona_variation=True,
                self_consistency=False
            )
        elif level == DiversityLevel.HIGH:
            return DiversityConfig(
                level=level,
                template_variation=True,
                context_injection=True,
                few_shot_learning=True,
                persona_variation=True,
                self_consistency=True,
                num_candidates=3
            )
        else:  # MAXIMUM
            return DiversityConfig(
                level=level,
                template_variation=True,
                context_injection=True,
                few_shot_learning=True,
                persona_variation=True,
                self_consistency=True,
                num_candidates=5,
                confidence_threshold=0.9
            )
    
    async def generate_single_item(self, category: str) -> Dict[str, Any]:
        """
        Generate a single item for the specified category.
        
        Args:
            category: Category identifier
            
        Returns:
            Dictionary with generation result
        """
        try:
            # Get prompts
            system_prompt = self.get_system_prompt()
            user_prompt = self.create_generation_prompt(category)
            
            # Make API call
            response_obj = await make_api_call_async(
                self.llm,
                system_prompt,
                user_prompt,
                response_schema=self.response_model,
                enable_thinking=self.enable_thinking,
                use_structure_model=self.use_structure_model
            )
            
            return {
                'success': True,
                'response': response_obj,
                'category': category,
                'synthesis_type': self.synthesis_type
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'category': category,
                'synthesis_type': self.synthesis_type
            }
    
    async def generate_batch_async(self, categories: List[str], concurrent_requests: int = 5, show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Generate multiple items asynchronously.
        
        Args:
            categories: List of categories to generate (one item per category)
            concurrent_requests: Number of concurrent API requests
            show_progress: Whether to show progress bar
            
        Returns:
            List of generation results
        """
        if show_progress:
            print(f"\nGenerating {len(categories)} {self.synthesis_type} items...")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def process_with_semaphore(category, index):
            async with semaphore:
                result = await self.generate_single_item(category)
                result['index'] = index
                return result
        
        # Process categories with progress bar
        tasks = [process_with_semaphore(cat, i) for i, cat in enumerate(categories)]
        
        results = []
        if show_progress:
            with tqdm(total=len(tasks), desc=f"Generating {self.synthesis_type}") as pbar:
                for i in range(0, len(tasks), concurrent_requests):
                    batch = tasks[i:i + concurrent_requests]
                    batch_results = await asyncio.gather(*batch)
                    results.extend(batch_results)
                    pbar.update(len(batch))
        else:
            # No progress bar
            for i in range(0, len(tasks), concurrent_requests):
                batch = tasks[i:i + concurrent_requests]
                batch_results = await asyncio.gather(*batch)
                results.extend(batch_results)
        
        return results
    
    def _create_item_record(self, index: int, response: BaseModel, category: str) -> Dict[str, Any]:
        """Create a record from the response model, adding metadata."""
        # Start with metadata
        record = {
            'id': index,
            'synthesis_type': self.synthesis_type,
            'category': category,
            'generation_timestamp': datetime.now().isoformat(),
            'timestamp': datetime.now().isoformat()  # For backward compatibility
        }
        
        # Add classification based on category
        category_info = self.prompts_manager.get_category_info(self.synthesis_type, category)
        record['classification'] = category_info.get('classification', 'UNKNOWN')
        
        # Add all fields from the LLM response
        for field_name, field_value in response.dict().items():
            record[field_name] = field_value
        
        return record
    
    def _create_error_record(self, index: int, category: str, error_message: str) -> Dict[str, Any]:
        """Create an error record for failed generations."""
        record = {
            'id': index,
            'synthesis_type': self.synthesis_type,
            'category': category,
            'generation_timestamp': datetime.now().isoformat(),
            'error': error_message,
            'success': False
        }
        
        # Add placeholder values for LLM schema fields
        llm_schema_def = self.prompts_manager.get_llm_response_schema(self.synthesis_type)
        for field_name in llm_schema_def:
            record[field_name] = None
        
        # Add classification placeholder
        record['classification'] = 'ERROR'
        
        return record
    
    def save_results(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Save generation results to files and optionally to MongoDB.
        
        Args:
            results: List of generation results
            
        Returns:
            Dictionary with file paths and save statistics
        """
        # Create output directory
        results_dir = self._create_results_directory()
        
        # Process results
        successful_results, error_results = self._separate_results(results)
        
        # Save to files
        save_paths = self._save_to_files(results_dir, successful_results, error_results, results)
        
        # Save to MongoDB if enabled
        if self.save_to_mongodb and successful_results:
            mongodb_result = self._save_to_mongodb(successful_results)
            save_paths['mongodb_result'] = mongodb_result
        elif self.save_to_mongodb and not successful_results:
            logger.warning("No successful results to save to MongoDB")
        
        return save_paths
    
    def _create_results_directory(self) -> Path:
        """Create timestamped results directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.output_dir / self.synthesis_type / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    def _separate_results(self, results: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Separate successful and error results."""
        successful_results = []
        error_results = []
        
        for result in results:
            if result['success']:
                record = self._create_item_record(
                    result['index'],
                    result['response'],
                    result['category']
                )
                successful_results.append(record)
            else:
                record = self._create_error_record(
                    result['index'],
                    result['category'],
                    result.get('error', 'Unknown error')
                )
                error_results.append(record)
        
        return successful_results, error_results
    
    def _save_to_files(self, results_dir: Path, successful_results: List[Dict], 
                       error_results: List[Dict], all_results: List[Dict]) -> Dict[str, Any]:
        """Save results to CSV and JSON files."""
        save_paths = {}
        
        # Save detailed results CSV
        combined_results = successful_results + error_results
        if combined_results:
            detailed_df = pd.DataFrame(combined_results)
            detailed_path = results_dir / "synthesis_results.csv"
            detailed_df.to_csv(detailed_path, index=False)
            save_paths['detailed_results'] = str(detailed_path)
        
        # Generate and save summary
        summary_stats = self._generate_summary_stats(
            all_results, successful_results, error_results, 
            detailed_df if combined_results else None
        )
        
        summary_path = results_dir / "synthesis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        save_paths['summary_json'] = str(summary_path)
        
        # Save human-readable report
        report_path = results_dir / "synthesis_report.txt"
        self._write_report(report_path, summary_stats, successful_results, error_results)
        save_paths['summary_report'] = str(report_path)
        
        # Add counts to return data
        save_paths.update({
            'success_count': len(successful_results),
            'error_count': len(error_results),
            'total_count': len(all_results)
        })
        
        return save_paths
    
    def _generate_summary_stats(self, all_results: List[Dict], successful_results: List[Dict], 
                               error_results: List[Dict], detailed_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate summary statistics for the results."""
        type_info = self.prompts_manager.get_synthesis_type_info(self.synthesis_type)
        
        stats = {
            'synthesis_type': self.synthesis_type,
            'synthesis_name': type_info.get('name', self.synthesis_type),
            'total_generated': len(all_results),
            'successful': len(successful_results),
            'errors': len(error_results),
            'success_rate': len(successful_results) / len(all_results) if all_results else 0,
            'provider': self.provider,
            'model': self.model,
            'generation_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'config': {
                'enable_thinking': self.enable_thinking,
                'use_structure_model': self.use_structure_model,
                'category': self.category
            }
        }
        
        # Add distribution statistics if we have a dataframe
        if detailed_df is not None and not detailed_df.empty:
            if 'category' in detailed_df.columns:
                stats['category_distribution'] = detailed_df['category'].value_counts().to_dict()
            if 'classification' in detailed_df.columns:
                stats['classification_distribution'] = detailed_df['classification'].value_counts().to_dict()
        
        return stats
    
    def _write_report(self, report_path: Path, summary_stats: Dict[str, Any], 
                      successful_results: List[Dict], error_results: List[Dict]):
        """Write a human-readable report of the generation results."""
        with open(report_path, 'w') as f:
            f.write(f"SYNTHESIS GENERATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Synthesis Type: {summary_stats['synthesis_name']}\n")
            f.write(f"Total generated: {summary_stats['total_generated']}\n")
            f.write(f"Successful: {summary_stats['successful']}\n")
            f.write(f"Errors: {summary_stats['errors']}\n")
            f.write(f"Success rate: {summary_stats['success_rate']:.2%}\n\n")
            
            # Category distribution
            if 'category_distribution' in summary_stats:
                f.write("Category Distribution:\n")
                for category, count in summary_stats['category_distribution'].items():
                    category_info = self.prompts_manager.get_category_info(self.synthesis_type, category)
                    category_name = category_info.get('name', category)
                    f.write(f"  {category_name}: {count}\n")
            
            # Classification distribution
            if 'classification_distribution' in summary_stats:
                f.write("\nClassification Distribution:\n")
                for classification, count in summary_stats['classification_distribution'].items():
                    f.write(f"  {classification}: {count}\n")
            
            f.write(f"\nGeneration completed: {summary_stats['generation_timestamp']}\n")
            f.write(f"Model: {self.provider} - {self.model}\n")
    
    def _save_to_mongodb(self, successful_results: List[Dict]) -> Dict[str, Any]:
        """Save results to MongoDB."""
        try:
            logger.info("Saving data to MongoDB...")
            
            db_service = get_scam_data_service()
            mongodb_result = db_service.store_synthesis_results(
                synthesis_type=self.synthesis_type,
                results=successful_results
            )
            
            if mongodb_result.get('success'):
                logger.info(f"Successfully saved {mongodb_result['inserted_count']} records to MongoDB")
            else:
                logger.error(f"Failed to save to MongoDB: {mongodb_result.get('error', 'Unknown error')}")
                
            return mongodb_result
            
        except ImportError:
            logger.error("Could not import database service")
            return {'success': False, 'error': 'Database service not available'}
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_checkpoint_filename(self) -> str:
        """Generate checkpoint filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.synthesis_type}_{self.provider}_{self.model}_{timestamp}.json"
    
    def _find_existing_checkpoint(self, checkpoint_dir: str = "checkpoints/synthesis") -> Optional[Path]:
        """Find existing checkpoint files."""
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None
        
        pattern = f"{self.synthesis_type}_{self.provider}_{self.model}_*.json"
        checkpoint_files = list(checkpoint_path.glob(pattern))
        
        # Also check for old transcript_generation checkpoints if synthesis_type is phone_transcript
        if self.synthesis_type == "phone_transcript" and not checkpoint_files:
            old_pattern = "transcript_generation_*.json"
            checkpoint_files = list(checkpoint_path.glob(old_pattern))
        
        if not checkpoint_files:
            return None
        
        # Return most recent checkpoint
        return max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    
    def load_checkpoint(self, checkpoint_dir: str = "checkpoints/synthesis",
                       resume_from_checkpoint: bool = True,
                       override_compatibility: bool = False) -> bool:
        """Load checkpoint if available."""
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
                if (checkpoint_data.get('synthesis_type') != self.synthesis_type or
                    checkpoint_data.get('provider') != self.provider or
                    checkpoint_data.get('model') != self.model):
                    print("Checkpoint configuration doesn't match current configuration.")
                    print("Use override_compatibility=True to force resume.")
                    return False
            
            # Load checkpoint data
            self.generated_items = checkpoint_data.get('generated_items', [])
            self.current_index = len(self.generated_items)
            self.checkpoint_file = str(checkpoint_file)
            
            print(f"Resumed from checkpoint: {checkpoint_file}")
            print(f"Already processed: {self.current_index} items")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def save_checkpoint(self, checkpoint_dir: str = "checkpoints/synthesis"):
        """Save current progress to checkpoint."""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Convert generated items to serializable format
        serializable_items = []
        for item in self.generated_items:
            if item['success'] and 'response' in item and item['response'] is not None:
                # Convert Pydantic model to dict with proper serialization
                serializable_item = item.copy()
                # Use model_dump for Pydantic v2 or dict for v1
                if hasattr(item['response'], 'model_dump'):
                    serializable_item['response'] = item['response'].model_dump(mode='json')
                elif hasattr(item['response'], 'dict'):
                    serializable_item['response'] = item['response'].dict()
                else:
                    # If it's already a dict or other type
                    serializable_item['response'] = item['response']
                serializable_items.append(serializable_item)
            else:
                serializable_items.append(item)
        
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'synthesis_type': self.synthesis_type,
            'provider': self.provider,
            'model': self.model,
            'current_index': self.current_index,
            'total_records': self.sample_size,  # Use total_records for consistency
            'generated_items': serializable_items,
            'progress': self.current_index / self.sample_size if self.sample_size > 0 else 0,
            'task': 'synthesis'  # Add task field for main.py compatibility
        }
        
        if not self.checkpoint_file:
            self.checkpoint_file = str(checkpoint_path / self._get_checkpoint_filename())
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            # Try alternative serialization
            def json_encoder(obj):
                if isinstance(obj, Enum):
                    return obj.value
                elif hasattr(obj, 'dict'):
                    return obj.dict()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                return str(obj)
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=json_encoder)
        
        tqdm.write(f"📁 Checkpoint saved: {self.current_index}/{self.sample_size}")
    
    async def process_full_generation_with_checkpoints(self,
                                                     checkpoint_interval: int = 100,
                                                     checkpoint_dir: str = "checkpoints/synthesis",
                                                     resume_from_checkpoint: bool = True,
                                                     concurrent_requests: int = 5,
                                                     override_compatibility: bool = False) -> Dict[str, Any]:
        """
        Generate items with checkpointing support.
        
        Args:
            checkpoint_interval: Save checkpoint every N items
            checkpoint_dir: Directory for checkpoint files
            resume_from_checkpoint: Whether to resume from existing checkpoint
            concurrent_requests: Number of concurrent API requests
            override_compatibility: Whether to override checkpoint compatibility check
            
        Returns:
            Dictionary with generation results
        """
        # Try to resume from checkpoint
        if resume_from_checkpoint:
            self.load_checkpoint(checkpoint_dir, resume_from_checkpoint, override_compatibility)
        
        if self.current_index >= self.sample_size:
            print("Generation already completed!")
            return {'status': 'completed', 'items': self.generated_items}
        
        # Setup models
        self.setup_models()
        
        # Get categories based on user selection
        if self.category == "ALL":
            # Get all available categories
            categories = self.prompts_manager.get_category_names(self.synthesis_type)
            if not categories:
                raise UnknownSynthesisTypeError(f"No categories defined for synthesis type: {self.synthesis_type}")
        else:
            # Use specific category only
            categories = [self.category]
        
        # Calculate remaining work
        remaining_count = self.sample_size - self.current_index
        print(f"Generating {remaining_count} additional {self.synthesis_type} items...")
        if self.category != "ALL":
            print(f"Using category: {self.category}")
        
        # Create category list for remaining items
        remaining_categories = []
        if self.category == "ALL":
            # Round-robin through all categories
            for i in range(remaining_count):
                category_index = (self.current_index + i) % len(categories)
                remaining_categories.append(categories[category_index])
            
            # Warn if starting fresh and not all categories will be used
            if self.current_index == 0 and self.sample_size < len(categories):
                print(f"\n⚠️  WARNING: Generating {self.sample_size} items but {len(categories)} categories available.")
                print(f"   Only the first {self.sample_size} categories will be used.")
                print(f"   Consider generating at least {len(categories)} items for full coverage.\n")
        else:
            # Use the same specific category for all items
            remaining_categories = [self.category] * remaining_count
        
        # Process in batches with checkpointing
        batch_size = max(1, checkpoint_interval)
        all_results = []
        
        with tqdm(total=remaining_count, desc=f"Generating {self.synthesis_type}") as pbar:
            for i in range(0, remaining_count, batch_size):
                batch_categories = remaining_categories[i:i + batch_size]
                
                # Generate batch (disable inner progress bar)
                batch_results = await self.generate_batch_async(batch_categories, concurrent_requests, show_progress=False)
                
                # Update indices
                for j, result in enumerate(batch_results):
                    result['index'] = self.current_index + j
                
                all_results.extend(batch_results)
                self.generated_items.extend(batch_results)
                self.current_index += len(batch_results)
                
                # Update progress
                pbar.update(len(batch_results))
                
                # Save checkpoint
                if (i + batch_size) % checkpoint_interval == 0 or i + batch_size >= remaining_count:
                    self.save_checkpoint(checkpoint_dir)
        
        # Save final results
        save_results = self.save_results(all_results)
        
        return {
            'status': 'completed',
            'results': save_results,
            'total_generated': len(all_results),
            'checkpoint_file': self.checkpoint_file
        }
    
    async def run_full_generation(self) -> Dict[str, Any]:
        """Run the complete generation process without checkpointing."""
        print("="*80)
        print(f"{self.synthesis_type.upper()} GENERATION PIPELINE")
        print("="*80)
        
        try:
            # Setup models
            self.setup_models()
            
            # Get categories based on user selection
            if self.category == "ALL":
                # Get all available categories
                categories = self.prompts_manager.get_category_names(self.synthesis_type)
                if not categories:
                    raise UnknownSynthesisTypeError(f"No categories defined for synthesis type: {self.synthesis_type}")
                    
                # Create category list (round-robin)
                generation_categories = []
                for i in range(self.sample_size):
                    category_index = i % len(categories)
                    generation_categories.append(categories[category_index])
                
                # Warn if not all categories will be used
                if self.sample_size < len(categories):
                    print(f"\n⚠️  WARNING: Generating {self.sample_size} items but {len(categories)} categories available.")
                    print(f"   Only categories {categories[:self.sample_size]} will be used.")
                    print(f"   Consider generating at least {len(categories)} items for full coverage.\n")
            else:
                # Use specific category for all items
                generation_categories = [self.category] * self.sample_size
                print(f"Generating all items with category: {self.category}")
            
            # Generate items
            results = await self.generate_batch_async(generation_categories)
            
            # Save results
            save_results = self.save_results(results)
            
            type_info = self.prompts_manager.get_synthesis_type_info(self.synthesis_type)
            print(f"\nGeneration completed successfully!")
            print(f"Generated: {save_results['success_count']} {type_info.get('name', self.synthesis_type)} items")
            print(f"Errors: {save_results['error_count']} items")
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