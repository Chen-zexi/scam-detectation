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

# Add current directory to Python path to allow imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from api_provider import LLM
from api_call import make_api_call
from data_loader import DatasetLoader
from results_saver import ResultsSaver
from pydantic import BaseModel

class AnnotationResponseSchema(BaseModel):
    explanation: str
    key_indicators: List[str]
    confidence: str  # "high", "medium", "low"

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

Guidelines:
- Be thorough but concise in your explanations
- Focus on actionable insights that help with detection
- For scams: explain the deception techniques, red flags, and malicious intent
- For legitimate content: explain trust indicators and why it's genuine
- Always be educational and help build detection skills

Respond with:
- explanation: A detailed explanation of why this content has the given classification
- key_indicators: A list of specific indicators/features that support the classification
- confidence: Your confidence level in the explanation ("high", "medium", "low")
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
                 output_dir: str = "annotations"):
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
        """
        self.dataset_path = dataset_path
        self.provider = provider
        self.model = model
        self.sample_size = sample_size
        self.balanced_sample = balanced_sample
        self.random_state = random_state
        self.content_columns = content_columns
        self.output_dir = output_dir
        
        # Initialize components
        self.data_loader = DatasetLoader(dataset_path)
        self.llm_instance = None
        self.llm = None
        self.prompt_generator = None
        self.annotations = []
        
    def setup_llm(self):
        """Initialize the LLM"""
        try:
            self.llm_instance = LLM(provider=self.provider, model=self.model)
            self.llm = self.llm_instance.get_llm()
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
                # Make API call with custom structured output for annotations
                prompt_template = self._get_annotation_prompt_template()
                messages = prompt_template.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
                client = self.llm.with_structured_output(AnnotationResponseSchema)
                response = client.invoke(messages)
                
                # Create comprehensive annotation record
                annotation = self._create_annotation_record(i+1, row, response)
                annotations.append(annotation)
                
                actual_class = 'Scam' if row['label'] == 1 else 'Legitimate'
                print(f"  Class: {actual_class}")
                print(f"  Confidence: {response.confidence}")
                print(f"  Key indicators: {len(response.key_indicators)}")
                print(f"  Explanation length: {len(response.explanation)} chars")
                
            except Exception as e:
                print(f"  Error annotating record {i+1}: {e}")
                annotation = self._create_error_annotation_record(i+1, row, str(e))
                annotations.append(annotation)
        
        self.annotations = annotations
        print(f"\nAnnotation completed. Processed {len(annotations)} records.")
        return annotations
    
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
            output_dir=args.output_dir
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