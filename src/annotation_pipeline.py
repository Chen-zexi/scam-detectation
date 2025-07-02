#!/usr/bin/env python3
"""
LLM Annotation Pipeline for Scam Detection

This pipeline uses an LLM to annotate data by providing explanations for why content
is labeled as legitimate or scam. The LLM is given both the original content and 
the correct answer, then asked to explain the reasoning behind the classification.

Usage:
    python annotation_pipeline.py --dataset path/to/dataset.csv --provider openai --model gpt-4 --sample-size 100
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import asyncio
import time
import json
from datetime import datetime
from tqdm import tqdm

# Add current directory to Python path to allow imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from api_provider import LLM
from api_call import make_api_call, parse_structured_output, make_api_call_async, parse_structured_output_async
from data_loader import DatasetLoader
from results_saver import ResultsSaver
from pydantic import BaseModel

class AnnotationResponseSchema(BaseModel):
    explanation: str
    key_indicators: List[str]
    confidence: str  # "high", "medium", "low"
    usability: bool  # True if data is meaningful, False if not meaningful/nonsensical

class AnnotationPromptGenerator:
    """Generate prompts for annotation tasks"""
    
    def __init__(self, features: List[str], content_columns: List[str] = None):
        self.features = features
        self.content_columns = content_columns or features
    
    def get_system_prompt(self) -> str:
        """Generate system prompt for annotation"""
        return """
You are an expert cybersecurity analyst and educator specializing in scam detection.
Your task is to provide detailed educational explanations for why content is classified 
as either a scam or legitimate.

You will be given:
1. The original content (email, message, etc.)
2. The correct classification (scam or legitimate)

Your job is to explain WHY this classification is correct by:
- Identifying key indicators that support the classification
- Explaining the reasoning in an educational manner
- Highlighting specific techniques or patterns used (for scams) or trust signals (for legitimate content)
- Providing insights that would help someone learn to identify similar cases
- Assessing whether the content is meaningful and usable for analysis (**Do not include explanation for usability in your finalresponse**)

Guidelines:
- Be thorough but concise in your explanations
- Focus on actionable insights that help with detection
- For scams: explain the deception techniques, red flags, and malicious intent
- For legitimate content: explain trust indicators and why it's genuine
- Always be educational and help build detection skills
- Evaluate if the content makes sense and is meaningful for analysis

Respond with:
- explanation: A detailed explanation of why this content has the given classification
- key_indicators: A list of specific indicators/features that support the classification
- confidence: Your confidence level in the explanation ("high", "medium", "low")
- usability: True if the content is meaningful and makes sense for analysis, False if the content is nonsensical, corrupted, or not meaningful
"""
    
    def create_annotation_prompt(self, row: Dict[str, Any], correct_label: str) -> str:
        """Create annotation prompt with content and correct answer"""
        classification = "SCAM" if correct_label == "1" or correct_label == 1 else "LEGITIMATE"
        
        prompt_parts = [
            f"Please analyze the following content that is classified as {classification}.\n",
            "Provide a detailed explanation of why this classification is correct.\n\n",
            "CONTENT TO ANALYZE:\n"
        ]
        
        # Add content features
        for feature in self.content_columns:
            if feature in row:
                value = row.get(feature, "")
                if value and str(value).strip():
                    prompt_parts.append(f"{feature}: {str(value).strip()}\n")
        
        prompt_parts.append(f"\nCORRECT CLASSIFICATION: {classification}\n")
        prompt_parts.append("\nProvide your detailed explanation of why this classification is correct.")
        
        return "".join(prompt_parts)

class LLMAnnotationPipeline:
    """
    Main annotation pipeline that generates explanations for labeled data
    """
    
    def __init__(self, 
                 dataset_path: str,
                 provider: str,
                 model: str,
                 sample_size: int = 100,
                 balanced_sample: bool = False,
                 random_state: int = 42,
                 content_columns: Optional[List[str]] = None,
                 output_dir: str = "annotations",
                 enable_thinking: bool = False,
                 use_structure_model: bool = False):
        """
        Initialize the annotation pipeline
        
        Args:
            dataset_path: Path to the CSV dataset file
            provider: LLM provider (e.g., 'openai', 'anthropic', 'local')
            model: Model name
            sample_size: Number of samples to annotate
            balanced_sample: Whether to sample equal numbers of scam and legitimate messages
            random_state: Random seed for reproducibility
            content_columns: List of column names to use as content for annotation
            output_dir: Directory to save annotation results
            enable_thinking: Whether to enable thinking tokens in prompts
            use_structure_model: Whether to use a separate structure model for parsing
        """
        self.dataset_path = dataset_path
        self.provider = provider
        self.model = model
        self.sample_size = sample_size
        self.balanced_sample = balanced_sample
        self.random_state = random_state
        self.content_columns = content_columns
        self.output_dir = output_dir
        self.enable_thinking = enable_thinking
        self.use_structure_model = use_structure_model
        
        # Initialize components
        self.data_loader = DatasetLoader(dataset_path)
        self.llm_instance = None
        self.llm = None
        self.structure_model = None
        self.prompt_generator = None
        self.annotations = []
        
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
                self.structure_model = self.llm_instance.get_structure_model(provider='lm-studio')
            print(f"LLM initialized successfully: {self.provider} - {self.model}")
        except Exception as e:
            raise Exception(f"Error initializing LLM: {e}")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load dataset and prepare sample for annotation"""
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
        
        # Initialize prompt generator
        self.prompt_generator = AnnotationPromptGenerator(self.data_loader.features, self.content_columns)
        
        print(f"\nDataset: {self.data_loader.dataset_name}")
        print(f"Sample size: {len(sample_df)} records")
        print(f"Content features: {', '.join(self.content_columns)}")
        
        return sample_df
    
    def annotate_sample(self, sample_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Annotate the sample dataset using the LLM
        
        Args:
            sample_df: Sample dataframe to annotate
            
        Returns:
            List of annotation results
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Call setup_llm() first.")
        
        if self.prompt_generator is None:
            raise ValueError("Prompt generator not initialized. Call load_and_prepare_data() first.")
        
        annotations = []
        system_prompt = self.prompt_generator.get_system_prompt()
        
        print("\n" + "="*80)
        print("STARTING LLM ANNOTATION PROCESS")
        print("="*80)
        
        for i, (_, row) in enumerate(sample_df.iterrows()):
            print(f"\nAnnotating record {i+1}/{len(sample_df)}...")
            
            # Create annotation prompt with correct label
            user_prompt = self.prompt_generator.create_annotation_prompt(row.to_dict(), str(row['label']))
            
            try:
                if self.use_structure_model:
                    # Make API call
                    response = make_api_call(self.llm, system_prompt, user_prompt, enable_thinking=self.enable_thinking, structure_model=True)
                    # Remove thinking tokens if they exist
                    if self.enable_thinking or ('<think>' in response and '</think>' in response):
                        import re
                        # Remove everything between <think> and </think> tags
                        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                    response = parse_structured_output(self.structure_model, response, AnnotationResponseSchema)
                else:
                    # Make API call with custom structured output for annotations
                    prompt_template = self._get_annotation_prompt_template()
                    messages = prompt_template.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
                    client = self.llm.with_structured_output(AnnotationResponseSchema)
                    response = client.invoke(messages)
                
                # Create comprehensive annotation record
                annotation = self._create_annotation_record(i+1, row, response)
                annotations.append(annotation)
                
                actual_class = 'Scam' if row['label'] == 1 else 'Legitimate'
                usability_status = 'Usable' if response.usability == 1 else 'Not usable'
                print(f"Class: {actual_class}")
                print(f"Confidence: {response.confidence}")
                print(f"Usability: {usability_status}")
                print(f"Key indicators: {len(response.key_indicators)}")
                print(f"Explanation length: {len(response.explanation)} chars")
                
            except Exception as e:
                print(f"  Error annotating record {i+1}: {e}")
                annotation = self._create_error_annotation_record(i+1, row, str(e))
                annotations.append(annotation)
        
        self.annotations = annotations
        print(f"\nAnnotation completed. Processed {len(annotations)} records.")
        return annotations

    async def annotate_sample_async(self, sample_df: pd.DataFrame, concurrent_requests: int = 10) -> List[Dict[str, Any]]:
        """
        Annotate the sample dataset using the LLM with concurrent requests
        
        Args:
            sample_df: Sample dataframe to annotate
            concurrent_requests: Number of concurrent requests to make (default: 10)
            
        Returns:
            List of annotation results
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Call setup_llm() first.")
        
        if self.prompt_generator is None:
            raise ValueError("Prompt generator not initialized. Call load_and_prepare_data() first.")
        
        system_prompt = self.prompt_generator.get_system_prompt()
        
        print("\n" + "="*80)
        print("STARTING LLM ANNOTATION PROCESS (ASYNC)")
        print(f"Concurrent requests: {concurrent_requests}")
        print("="*80)
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def annotate_single_record(i: int, row: pd.Series) -> Dict[str, Any]:
            async with semaphore:
                print(f"Annotating record {i+1}/{len(sample_df)}...")
                
                # Create annotation prompt with correct label
                user_prompt = self.prompt_generator.create_annotation_prompt(row.to_dict(), str(row['label']))
                
                try:
                    if self.use_structure_model:
                        # Make async API call
                        response = await make_api_call_async(self.llm, system_prompt, user_prompt, 
                                                           enable_thinking=self.enable_thinking, structure_model=True)
                        # Remove thinking tokens if they exist
                        if self.enable_thinking or ('<think>' in response and '</think>' in response):
                            import re
                            # Remove everything between <think> and </think> tags
                            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                        response = await parse_structured_output_async(self.structure_model, response, AnnotationResponseSchema)
                    else:
                        # Make async API call with custom structured output for annotations
                        prompt_template = self._get_annotation_prompt_template()
                        messages = prompt_template.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
                        client = self.llm.with_structured_output(AnnotationResponseSchema)
                        response = await client.ainvoke(messages)
                    
                    # Create comprehensive annotation record
                    annotation = self._create_annotation_record(i+1, row, response)
                    
                    actual_class = 'Scam' if row['label'] == 1 else 'Legitimate'
                    usability_status = 'Usable' if response.usability == 1 else 'Not usable'
                    print(f"Record {i+1} - Class: {actual_class}, Confidence: {response.confidence}, "
                          f"Usability: {usability_status}, Indicators: {len(response.key_indicators)}")
                    
                    return annotation
                    
                except Exception as e:
                    print(f"  Error annotating record {i+1}: {e}")
                    annotation = self._create_error_annotation_record(i+1, row, str(e))
                    return annotation
        
        # Create tasks for all records
        tasks = []
        for i, (_, row) in enumerate(sample_df.iterrows()):
            task = annotate_single_record(i, row)
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        annotations = await asyncio.gather(*tasks, return_exceptions=False)
        end_time = time.time()
        
        # Handle any exceptions that might have been returned
        valid_annotations = []
        for i, annotation in enumerate(annotations):
            if isinstance(annotation, Exception):
                print(f"Exception in record {i+1}: {annotation}")
                # Create error record
                row = sample_df.iloc[i]
                error_annotation = self._create_error_annotation_record(i+1, row, str(annotation))
                valid_annotations.append(error_annotation)
            else:
                valid_annotations.append(annotation)
        
        self.annotations = valid_annotations
        total_time = end_time - start_time
        print(f"\nAsync annotation completed in {total_time:.2f} seconds. Processed {len(valid_annotations)} records.")
        print(f"Average time per record: {total_time/len(valid_annotations):.2f} seconds")
        return valid_annotations
    
    def _get_annotation_prompt_template(self):
        """Get prompt template for annotations"""
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate([
            ("system", "{system_prompt}"),
            ("user", "{user_prompt}")
        ])
    
    def _create_annotation_record(self, 
                                 record_id: int, 
                                 row: pd.Series, 
                                 response: AnnotationResponseSchema) -> Dict[str, Any]:
        """Create a comprehensive annotation record"""
        annotation = {
            'record_id': record_id,
            'label': row['label'],
            'class': 'Scam' if row['label'] == 1 else 'Legitimate',
            'explanation': response.explanation,
            'key_indicators': response.key_indicators,
            'confidence': response.confidence,
            'usability': response.usability,
            'annotation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Add id column if it exists
        if 'id' in row:
            annotation['id'] = row['id']
        
        # Add all original features except id (to avoid duplication)
        for feature in self.data_loader.features:
            if feature != 'id':
                annotation[f'original_{feature}'] = row[feature]
        
        return annotation
    
    def _create_error_annotation_record(self, record_id: int, row: pd.Series, error_message: str) -> Dict[str, Any]:
        """Create an error annotation record"""
        annotation = {
            'record_id': record_id,
            'label': row['label'],
            'class': 'Scam' if row['label'] == 1 else 'Legitimate',
            'explanation': f'Error during annotation: {error_message}',
            'key_indicators': [],
            'confidence': 'error',
            'usability': 0,  # Default to not usable for error cases
            'annotation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Add id column if it exists
        if 'id' in row:
            annotation['id'] = row['id']
        
        # Add all original features except id (to avoid duplication)
        for feature in self.data_loader.features:
            if feature != 'id':
                annotation[f'original_{feature}'] = row[feature]
        
        return annotation
    
    def save_annotations(self) -> Dict[str, str]:
        """Save annotation results in result/annotated/{dataset_name}/{timestamp}/ structure"""
        if not self.annotations:
            raise ValueError("No annotations available. Run annotation first.")
        
        # Create directory structure: result/annotated/{dataset_name}/{timestamp}/
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.output_dir) / self.data_loader.dataset_name / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save annotated dataset
        annotations_df = pd.DataFrame(self.annotations)
        annotations_path = results_dir / f"{self.data_loader.dataset_name}_annotated.csv"
        annotations_df.to_csv(annotations_path, index=False)
        
        # Create and save annotation summary
        summary_path = self._save_annotation_summary(results_dir)
        
        print(f"\nAnnotation results saved to: {results_dir}")
        print(f"- Annotated dataset: {annotations_path}")
        print(f"- Summary: {summary_path}")
        
        return {
            'results_directory': str(results_dir),
            'annotated_dataset_path': str(annotations_path),
            'summary_path': str(summary_path)
        }
    
    def _save_annotation_summary(self, results_dir: Path) -> Path:
        """Create and save annotation summary"""
        if not self.annotations:
            return results_dir / "annotation_summary.json"
        
        # Calculate annotation statistics
        total_annotations = len(self.annotations)
        successful_annotations = len([a for a in self.annotations if a['confidence'] != 'error'])
        error_count = total_annotations - successful_annotations
        
        # Confidence distribution
        confidence_dist = {}
        for annotation in self.annotations:
            conf = annotation['confidence']
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
        
        # Class distribution
        class_dist = {}
        for annotation in self.annotations:
            cls = annotation['class']
            class_dist[cls] = class_dist.get(cls, 0) + 1
        
        # Usability distribution
        usability_dist = {}
        for annotation in self.annotations:
            usability = annotation.get('usability', 0)
            usability_key = 'Usable' if usability == 1 else 'Not usable'
            usability_dist[usability_key] = usability_dist.get(usability_key, 0) + 1
        
        # Average indicators per class
        scam_indicators = [len(a['key_indicators']) for a in self.annotations if a['class'] == 'Scam' and a['confidence'] != 'error']
        legit_indicators = [len(a['key_indicators']) for a in self.annotations if a['class'] == 'Legitimate' and a['confidence'] != 'error']
        
        # Get dataset info
        dataset_info = self.data_loader.get_dataset_info()
        
        summary = {
            'annotation_info': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'dataset_name': self.data_loader.dataset_name,
                'dataset_path': self.dataset_path,
                'model_provider': self.provider,
                'model_name': self.model,
                'sample_size_requested': self.sample_size,
                'balanced_sample': self.balanced_sample,
                'random_state': self.random_state,
                'content_columns': self.content_columns
            },
            'dataset_info': {
                'total_records_in_dataset': dataset_info['total_records'],
                'scam_count_in_dataset': dataset_info['scam_count'],
                'legitimate_count_in_dataset': dataset_info['legitimate_count'],
                'all_features': dataset_info['features']
            },
            'annotation_results': {
                'total_annotations': total_annotations,
                'successful_annotations': successful_annotations,
                'error_count': error_count,
                'success_rate': successful_annotations / total_annotations if total_annotations > 0 else 0
            },
            'confidence_distribution': confidence_dist,
            'class_distribution': class_dist,
            'usability_distribution': usability_dist,
            'indicator_statistics': {
                'avg_indicators_scam': sum(scam_indicators) / len(scam_indicators) if scam_indicators else 0,
                'avg_indicators_legitimate': sum(legit_indicators) / len(legit_indicators) if legit_indicators else 0,
                'max_indicators': max([len(a['key_indicators']) for a in self.annotations if a['confidence'] != 'error'], default=0),
                'min_indicators': min([len(a['key_indicators']) for a in self.annotations if a['confidence'] != 'error'], default=0)
            }
        }
        
        # Save summary
        import json
        summary_path = results_dir / "annotation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nAnnotation Summary:")
        print(f"- Total records: {total_annotations}")
        print(f"- Successful annotations: {successful_annotations}")
        print(f"- Success rate: {summary['annotation_results']['success_rate']:.2%}")
        print(f"- Usable records: {usability_dist.get('Usable', 0)}")
        print(f"- Not usable records: {usability_dist.get('Not usable', 0)}")
        print(f"- Usability rate: {usability_dist.get('Usable', 0) / total_annotations:.2%}" if total_annotations > 0 else "- Usability rate: 0%")
        
        return summary_path
    
    def run_full_annotation(self) -> Dict[str, Any]:
        """Run the complete annotation pipeline"""
        print("Starting LLM Annotation Pipeline...")
        
        # Setup LLM
        self.setup_llm()
        
        # Load and prepare data
        sample_df = self.load_and_prepare_data()
        
        # Run annotation
        annotations = self.annotate_sample(sample_df)
        
        # Save results
        save_paths = self.save_annotations()
        
        return {
            'annotations': annotations,
            'save_paths': save_paths,
            'summary': {
                'total_records': len(annotations),
                'successful_annotations': len([a for a in annotations if a['confidence'] != 'error']),
                'dataset_name': self.data_loader.dataset_name
            }
        }

    async def run_full_annotation_async(self, concurrent_requests: int = 10) -> Dict[str, Any]:
        """Run the complete annotation pipeline asynchronously"""
        print("Starting LLM Annotation Pipeline (Async)...")
        
        # Setup LLM
        self.setup_llm()
        
        # Load and prepare data
        sample_df = self.load_and_prepare_data()
        
        # Run annotation asynchronously
        annotations = await self.annotate_sample_async(sample_df, concurrent_requests)
        
        # Save results
        save_paths = self.save_annotations()
        
        return {
            'annotations': annotations,
            'save_paths': save_paths,
            'summary': {
                'total_records': len(annotations),
                'successful_annotations': len([a for a in annotations if a['confidence'] != 'error']),
                'dataset_name': self.data_loader.dataset_name
            }
        }
    
    # ==================== CHECKPOINT FUNCTIONALITY ====================
    
    def _get_checkpoint_filename(self) -> str:
        """Generate checkpoint filename based on configuration"""
        dataset_name = Path(self.dataset_path).stem
        # Sanitize model name for filesystem compatibility
        safe_model_name = self.model.replace("/", "-").replace("\\", "-").replace(":", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{dataset_name}_annotation_{self.provider}_{safe_model_name}_{timestamp}.json"
    
    def _find_existing_checkpoint(self, checkpoint_dir: str = "checkpoints") -> Optional[Path]:
        """Find the most recent checkpoint file for this configuration"""
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None
            
        dataset_name = Path(self.dataset_path).stem
        # Sanitize model name for filesystem compatibility
        safe_model_name = self.model.replace("/", "-").replace("\\", "-").replace(":", "-")
        pattern = f"{dataset_name}_annotation_{self.provider}_{safe_model_name}_*.json"
        
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
            self.annotations = checkpoint_data.get('annotations', [])
            self.checkpoint_file = checkpoint_file
            
            print(f"Loaded checkpoint from {checkpoint_file}")
            print(f"Resuming from index {self.current_index} with {len(self.annotations)} existing annotations")
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
            'annotations': self.annotations,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def _write_checkpoint_message(self, message: str):
        """Write checkpoint message, clearing the previous one"""
        # Clear previous checkpoint message if it exists
        if self.last_checkpoint_message:
            # Move cursor up and clear line
            tqdm.write('\033[F\033[K', end='')
        
        # Write new checkpoint message
        tqdm.write(message)
        self.last_checkpoint_message = message
    
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
            Dictionary with annotation results and file paths
        """
        print("="*80)
        print("ANNOTATION PIPELINE WITH CHECKPOINTING")
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
        self.prompt_generator = AnnotationPromptGenerator(self.data_loader.features, self.content_columns)
        
        print(f"Dataset: {self.data_loader.dataset_name}")
        print(f"Total records: {self.total_records:,}")
        print(f"Checkpoint interval: {checkpoint_interval:,}")
        print(f"Content features: {', '.join(self.content_columns)}")
        
        # Load checkpoint if available
        self.load_checkpoint(checkpoint_dir, resume_from_checkpoint, override_compatibility)
        
        self.start_time = time.time()
        total_processed = len(self.annotations)
        
        print(f"\nProcessing records {self.current_index + 1} to {self.total_records}...")
        
        # Create progress bar
        progress_bar = tqdm(
            range(self.current_index, self.total_records),
            desc="Annotating",
            unit="records",
            initial=self.current_index,
            total=self.total_records
        )
        
        # Process remaining records
        for i in progress_bar:
            row = dataset_df.iloc[i]
            
            # Update progress bar description
            progress_bar.set_description(f"Annotating record {i + 1}/{self.total_records}")
            
            try:
                # Create annotation prompt
                system_prompt = self.prompt_generator.get_system_prompt()
                user_prompt = self.prompt_generator.create_annotation_prompt(row.to_dict(), str(row['label']))
                
                # Make API call
                if self.use_structure_model:
                    response = make_api_call(self.llm, system_prompt, user_prompt,
                                           enable_thinking=self.enable_thinking, structure_model=True)
                    if self.enable_thinking or ('<think>' in response and '</think>' in response):
                        import re
                        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                    response = parse_structured_output(self.structure_model, response, AnnotationResponseSchema)
                else:
                    # Make API call with custom structured output for annotations
                    prompt_template = self._get_annotation_prompt_template()
                    messages = prompt_template.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
                    client = self.llm.with_structured_output(AnnotationResponseSchema)
                    response = client.invoke(messages)
                
                # Create annotation record
                annotation = self._create_annotation_record(i, row, response)
                
            except Exception as e:
                tqdm.write(f"Error processing record {i + 1}: {e}")
                annotation = self._create_error_annotation_record(i, row, str(e))
            
            self.annotations.append(annotation)
            self.current_index = i + 1
            total_processed += 1
            
            # Save checkpoint at intervals
            if (i + 1) % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_dir)
                self._write_checkpoint_message(f"Checkpoint saved at record {i + 1}")
        
        # Close progress bar
        progress_bar.close()
        
        # Final checkpoint
        self.save_checkpoint(checkpoint_dir)
        
        # Save final results
        save_paths = self.save_annotations()
        
        total_time = time.time() - self.start_time
        print(f"\n{'='*80}")
        print("ANNOTATION COMPLETED")
        print(f"{'='*80}")
        print(f"Total records processed: {total_processed:,}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per record: {total_time/total_processed:.3f} seconds")
        
        return {
            'annotations': self.annotations,
            'save_paths': save_paths,
            'summary': {
                'total_records': len(self.annotations),
                'successful_annotations': len([a for a in self.annotations if a['confidence'] != 'error']),
                'dataset_name': self.data_loader.dataset_name,
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
            Dictionary with annotation results and file paths
        """
        print("="*80)
        print("ASYNC ANNOTATION PIPELINE WITH CHUNKED BATCHING")
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
        self.prompt_generator = AnnotationPromptGenerator(self.data_loader.features, self.content_columns)
        
        print(f"Dataset: {self.data_loader.dataset_name}")
        print(f"Total records: {self.total_records:,}")
        print(f"Checkpoint interval: {checkpoint_interval:,}")
        print(f"Content features: {', '.join(self.content_columns)}")
        
        # Load checkpoint if available
        self.load_checkpoint(checkpoint_dir, resume_from_checkpoint, override_compatibility)
        
        self.start_time = time.time()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
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
            async def annotate_with_semaphore(index: int, row: pd.Series):
                async with semaphore:
                    result = await self._annotate_single_record_async(index, row)
                    chunk_progress.update(1)
                    return result
            
            tasks = []
            for i in range(chunk_start, chunk_end):
                row = dataset_df.iloc[i]
                tasks.append(annotate_with_semaphore(i, row))
            
            # Process chunk
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for i, result in enumerate(chunk_results):
                actual_index = chunk_start + i
                if isinstance(result, Exception):
                    row = dataset_df.iloc[actual_index]
                    result = self._create_error_annotation_record(actual_index, row, str(result))
                
                self.annotations.append(result)
                self.current_index = actual_index + 1
                overall_progress.update(1)
            
            # Close chunk progress bar
            chunk_progress.close()
            
            # Save checkpoint after each chunk
            if chunk_end % checkpoint_interval <= chunk_size:
                self.save_checkpoint(checkpoint_dir)
                self._write_checkpoint_message(f"Checkpoint saved at record {chunk_end}")
        
        # Close overall progress bar
        overall_progress.close()
        
        # Final checkpoint
        self.save_checkpoint(checkpoint_dir)
        
        # Save final results
        save_paths = self.save_annotations()
        
        total_time = time.time() - self.start_time
        total_processed = len(self.annotations)
        
        print(f"\n{'='*80}")
        print("CHUNKED ASYNC ANNOTATION COMPLETED")
        print(f"{'='*80}")
        print(f"Total records processed: {total_processed:,}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per record: {total_time/total_processed:.3f} seconds")
        
        return {
            'annotations': self.annotations,
            'save_paths': save_paths,
            'summary': {
                'total_records': len(self.annotations),
                'successful_annotations': len([a for a in self.annotations if a['confidence'] != 'error']),
                'dataset_name': self.data_loader.dataset_name,
                'checkpoint_file': str(self.checkpoint_file) if self.checkpoint_file else None
            }
        }
    
    async def _annotate_single_record_async(self, index: int, row: pd.Series) -> Dict[str, Any]:
        """Annotate a single record asynchronously"""
        try:
            system_prompt = self.prompt_generator.get_system_prompt()
            user_prompt = self.prompt_generator.create_annotation_prompt(row.to_dict(), str(row['label']))
            
            if self.use_structure_model:
                response = await make_api_call_async(self.llm, system_prompt, user_prompt,
                                                   enable_thinking=self.enable_thinking, structure_model=True)
                if self.enable_thinking or ('<think>' in response and '</think>' in response):
                    import re
                    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                response = await parse_structured_output_async(self.structure_model, response, AnnotationResponseSchema)
            else:
                # Make async API call with custom structured output for annotations
                prompt_template = self._get_annotation_prompt_template()
                messages = prompt_template.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
                client = self.llm.with_structured_output(AnnotationResponseSchema)
                response = await client.ainvoke(messages)
            
            return self._create_annotation_record(index, row, response)
            
        except Exception as e:
            return self._create_error_annotation_record(index, row, str(e))

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
            Dictionary with annotation results and file paths
        """
        print("="*80)
        print("ASYNC ANNOTATION PIPELINE WITH OVERLAPPING BATCHES")
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
        self.prompt_generator = AnnotationPromptGenerator(self.data_loader.features, self.content_columns)
        
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
                return await self._annotate_single_record_async(index, row)
        
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
                        result = self._create_error_annotation_record(index, row, str(e))
                    
                    completed_results[index] = result
                    overall_progress.update(1)
                
                # Process completed results in order and add to final results
                while self.current_index in completed_results:
                    result = completed_results.pop(self.current_index)
                    self.annotations.append(result)
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
                            f"(session: {elapsed_session/60:.1f}m, rate: {records_processed_this_session/elapsed_session:.2f}r/s)"
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
        print(f"   Total records completed: {len(self.annotations):,}")
        
        if rates_history:
            print(f"   Recent rate: {rates_history[-1]:.2f} records/second")
            print(f"   Rate stability: {min(rates_history)/max(rates_history):.2f} (1.0 = perfectly stable)")
        
        # Final checkpoint
        self.save_checkpoint(checkpoint_dir)
        
        # Save final results
        save_paths = self.save_annotations()
        
        print(f"\n{'='*80}")
        print("ASYNC ANNOTATION COMPLETED")
        print(f"{'='*80}")
        print(f"Total records completed: {len(self.annotations):,}")
        print(f"Session processing rate: {final_rate:.2f} records/second")
        
        return {
            'annotations': self.annotations,
            'save_paths': save_paths,
            'summary': {
                'total_records': len(self.annotations),
                'successful_annotations': len([a for a in self.annotations if a['confidence'] != 'error']),
                'usable_annotations': len([a for a in self.annotations if a.get('usability', True)]),
                'success_rate': len([a for a in self.annotations if a['confidence'] != 'error']) / len(self.annotations),
                'usability_rate': len([a for a in self.annotations if a.get('usability', True)]) / len(self.annotations),
                'dataset_name': self.data_loader.dataset_name,
                'final_processing_rate': final_rate,
                'total_processing_time': total_time,
                'checkpoint_file': str(self.checkpoint_file) if self.checkpoint_file else None
            }
        }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LLM Annotation Pipeline for Scam Detection",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Path to the dataset CSV file (must contain a "label" column)'
    )
    
    parser.add_argument(
        '--provider', 
        type=str, 
        required=True,
        choices=['openai', 'anthropic', 'gemini', 'local'],
        help='LLM provider to use'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Model name to use (e.g., gpt-4, claude-3-sonnet, gemini-pro, etc.)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=100,
        help='Number of samples to annotate (default: 100)'
    )
    
    parser.add_argument(
        '--balanced-sample',
        action='store_true',
        help='Sample equal numbers of scam and legitimate messages'
    )
    
    parser.add_argument(
        '--random-state', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--content-columns',
        type=str,
        nargs='+',
        help='Specific columns to use as content (e.g., --content-columns subject body)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='annotations',
        help='Directory for output results (default: annotations)'
    )
    
    parser.add_argument(
        '--enable-thinking',
        action='store_true',
        help='Enable thinking tokens in prompts'
    )
    
    parser.add_argument(
        '--use-structure-model',
        action='store_true',
        help='Use a separate structure model for parsing responses'
    )
    
    return parser.parse_args()

def validate_dataset(dataset_path: str):
    """Validate that the dataset exists and has required structure"""
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    import pandas as pd
    try:
        df_sample = pd.read_csv(dataset_path, nrows=1)
        if 'label' not in df_sample.columns:
            raise ValueError("Dataset must contain a 'label' column")
        print(f"✓ Dataset validation passed")
        print(f"  - File: {dataset_path}")
        print(f"  - Columns: {list(df_sample.columns)}")
    except Exception as e:
        raise ValueError(f"Invalid dataset format: {e}")

def main():
    """Main annotation pipeline execution"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        print("="*80)
        print("LLM ANNOTATION PIPELINE FOR SCAM DETECTION")
        print("="*80)
        print(f"Dataset: {args.dataset}")
        print(f"Provider: {args.provider}")
        print(f"Model: {args.model}")
        print(f"Sample size: {args.sample_size}")
        print(f"Balanced sampling: {args.balanced_sample}")
        print(f"Random state: {args.random_state}")
        if args.content_columns:
            print(f"Content columns: {args.content_columns}")
        else:
            print(f"Content columns: All non-label columns")
        print(f"Output directory: {args.output_dir}")
        print(f"Enable thinking: {args.enable_thinking}")
        print(f"Use structure model: {args.use_structure_model}")
        
        # Validate dataset
        print(f"\nValidating dataset...")
        validate_dataset(args.dataset)
        
        # Initialize annotation pipeline
        pipeline = LLMAnnotationPipeline(
            dataset_path=args.dataset,
            provider=args.provider,
            model=args.model,
            sample_size=args.sample_size,
            balanced_sample=args.balanced_sample,
            random_state=args.random_state,
            content_columns=args.content_columns,
            output_dir=args.output_dir,
            enable_thinking=args.enable_thinking,
            use_structure_model=args.use_structure_model
        )
        
        # Run full annotation
        results = pipeline.run_full_annotation()
        
        # Print final summary
        print(f"\n{'='*80}")
        print("ANNOTATION COMPLETED")
        print(f"{'='*80}")
        
        save_paths = results['save_paths']
        summary = results['summary']
        
        print(f"Results directory: {save_paths['results_directory']}")
        print(f"Detailed annotations: {save_paths['detailed_results_path']}")
        print(f"Total records annotated: {summary['total_records']}")
        print(f"Successful annotations: {summary['successful_annotations']}")
        
        print(f"\n✓ Annotation pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nAnnotation interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 