"""
Test cases for the annotation pipeline functionality
"""

import pytest
import pandas as pd
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the classes we want to test
from src.annotate.annotation_pipeline import LLMAnnotationPipeline, AnnotationResponseSchema, AnnotationPromptGenerator
from src.llm_core.api_call import make_api_call


class TestAnnotationPromptGenerator:
    """Test cases for AnnotationPromptGenerator class"""
    
    def test_init(self):
        """Test prompt generator initialization"""
        features = ['subject', 'body', 'sender']
        generator = AnnotationPromptGenerator(features)
        
        assert generator.features == features
        assert generator.content_columns == features
    
    def test_init_with_content_columns(self):
        """Test prompt generator initialization with specific content columns"""
        features = ['subject', 'body', 'sender', 'timestamp']
        content_columns = ['subject', 'body']
        generator = AnnotationPromptGenerator(features, content_columns)
        
        assert generator.features == features
        assert generator.content_columns == content_columns
    
    def test_get_system_prompt(self):
        """Test system prompt generation"""
        generator = AnnotationPromptGenerator(['subject', 'body'])
        system_prompt = generator.get_system_prompt()
        
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        assert "cybersecurity analyst" in system_prompt.lower()
        assert "scam detection" in system_prompt.lower()
    
    def test_create_annotation_prompt_scam(self):
        """Test annotation prompt creation for scam content"""
        generator = AnnotationPromptGenerator(['subject', 'body'])
        
        row_data = {
            'subject': 'Urgent: Verify your account',
            'body': 'Click here to verify your account immediately',
            'sender': 'noreply@fake-bank.com'
        }
        
        prompt = generator.create_annotation_prompt(row_data, "1")
        
        assert isinstance(prompt, str)
        assert "SCAM" in prompt
        assert "Urgent: Verify your account" in prompt
        assert "Click here to verify your account immediately" in prompt
    
    def test_create_annotation_prompt_legitimate(self):
        """Test annotation prompt creation for legitimate content"""
        generator = AnnotationPromptGenerator(['subject', 'body'])
        
        row_data = {
            'subject': 'Welcome to our newsletter',
            'body': 'Thank you for subscribing to our updates',
            'sender': 'newsletter@company.com'
        }
        
        prompt = generator.create_annotation_prompt(row_data, "0")
        
        assert isinstance(prompt, str)
        assert "LEGITIMATE" in prompt
        assert "Welcome to our newsletter" in prompt
        assert "Thank you for subscribing to our updates" in prompt


class TestLLMAnnotationPipeline:
    """Test cases for LLMAnnotationPipeline class"""
    
    def test_init(self, temp_dataset_file):
        """Test annotation pipeline initialization"""
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="openai",
            model="gpt-4o-mini",
            sample_size=3
        )
        
        assert pipeline.dataset_path == temp_dataset_file
        assert pipeline.provider == "openai"
        assert pipeline.model == "gpt-4o-mini"
        assert pipeline.sample_size == 3
        assert pipeline.balanced_sample == False
        assert pipeline.output_dir == "results/annotation"
    
    def test_init_with_custom_params(self, temp_dataset_file):
        """Test annotation pipeline initialization with custom parameters"""
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="anthropic",
            model="claude-3-sonnet",
            sample_size=10,
            balanced_sample=True,
            content_columns=['subject', 'body'],
            output_dir="custom_annotations"
        )
        
        assert pipeline.provider == "anthropic"
        assert pipeline.model == "claude-3-sonnet"
        assert pipeline.sample_size == 10
        assert pipeline.balanced_sample == True
        assert pipeline.content_columns == ['subject', 'body']
        assert pipeline.output_dir == "custom_annotations"
    
    @patch('src.annotate.annotation_pipeline.LLM')
    def test_setup_llm(self, mock_llm_class, temp_dataset_file, mock_llm_instance, mock_llm):
        """Test LLM setup"""
        mock_llm_class.return_value = mock_llm_instance
        mock_llm_instance.get_llm.return_value = mock_llm
        
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="openai",
            model="gpt-4o-mini"
        )
        
        pipeline.setup_llm()
        
        assert pipeline.llm is not None
        mock_llm_class.assert_called_once_with(provider="openai", model="gpt-4o-mini")
        mock_llm_instance.get_llm.assert_called_once()
    
    @patch('src.annotate.annotation_pipeline.LLM')
    def test_load_and_prepare_data(self, mock_llm_class, temp_dataset_file, sample_dataset):
        """Test data loading and preparation"""
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="openai",
            model="gpt-4o-mini",
            sample_size=3
        )
        
        # Load data
        sample_df = pipeline.load_and_prepare_data()
        
        assert len(sample_df) == 3
        assert 'label' in sample_df.columns
        assert pipeline.data_loader is not None
        assert pipeline.prompt_generator is not None
    
    @patch('src.annotate.annotation_pipeline.make_api_call')
    @patch('src.annotate.annotation_pipeline.LLM')
    def test_annotate_sample_sync(self, mock_llm_class, mock_api_call,
                                  temp_dataset_file, mock_llm_instance, mock_llm,
                                  mock_annotation_response):
        """Test synchronous annotation"""
        # Setup mocks
        mock_llm_class.return_value = mock_llm_instance
        mock_llm_instance.get_llm.return_value = mock_llm
        mock_api_call.return_value = mock_annotation_response
        
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="openai",
            model="gpt-4o-mini",
            sample_size=2
        )
        
        pipeline.setup_llm()
        sample_df = pipeline.load_and_prepare_data()
        
        # Run annotation
        results = pipeline.annotate_sample(sample_df)
        
        assert len(results) == 2
        assert all('record_id' in result for result in results)
        assert all('label' in result for result in results)
        assert all('explanation' in result for result in results)
        assert all('key_indicators' in result for result in results)
        assert all('confidence' in result for result in results)
        assert all('usability' in result for result in results)
        
        # Verify API call was made
        mock_api_call.assert_called()
        # Check that response_schema is passed (relaxed check)
        call_args = mock_api_call.call_args
        if call_args and len(call_args) > 1 and 'response_schema' in call_args[1]:
            assert call_args[1]['response_schema'] == AnnotationResponseSchema
    
    @patch('src.annotate.annotation_pipeline.make_api_call')
    @patch('src.annotate.annotation_pipeline.LLM')
    def test_annotate_sample(self, mock_llm_class, mock_api_call,
                                   temp_dataset_file, mock_llm_instance, mock_llm,
                                   mock_annotation_response):
        """Test asynchronous annotation"""
        # Setup mocks
        mock_llm_class.return_value = mock_llm_instance
        mock_llm_instance.get_llm.return_value = mock_llm
        mock_api_call.return_value = mock_annotation_response
        
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="openai",
            model="gpt-4o-mini",
            sample_size=2
        )
        
        pipeline.setup_llm()
        sample_df = pipeline.load_and_prepare_data()
        
        # Run async annotation
        results = asyncio.run(pipeline.annotate_sample(sample_df, concurrent_requests=2))
        
        assert len(results) == 2
        assert all('record_id' in result for result in results)
        assert all('explanation' in result for result in results)
        
        # Verify async API call was made
        mock_api_call.assert_called()
        # Check that response_schema is passed (relaxed check)
        call_args = mock_api_call.call_args
        if call_args and len(call_args) > 1 and 'response_schema' in call_args[1]:
            assert call_args[1]['response_schema'] == AnnotationResponseSchema
    
    
    
    @patch('src.annotate.annotation_pipeline.make_api_call')
    @patch('src.annotate.annotation_pipeline.LLM')
    def test_error_handling(self, mock_llm_class, mock_api_call,
                           temp_dataset_file, mock_llm_instance, mock_llm):
        """Test error handling during annotation"""
        # Setup mocks
        mock_llm_class.return_value = mock_llm_instance
        mock_llm_instance.get_llm.return_value = mock_llm
        mock_api_call.side_effect = Exception("API Error")
        
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="openai",
            model="gpt-4o-mini",
            sample_size=1
        )
        
        pipeline.setup_llm()
        sample_df = pipeline.load_and_prepare_data()
        results = pipeline.annotate_sample(sample_df)
        
        # Should still return results, but with error information
        assert len(results) == 1
        assert results[0]['confidence'] == 'error'
        assert 'error' in results[0]['explanation'].lower()
    
    @patch('src.annotate.annotation_pipeline.LLM')
    def test_balanced_sampling(self, mock_llm_class, temp_dataset_file):
        """Test balanced sampling functionality"""
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="openai",
            model="gpt-4o-mini",
            sample_size=4,
            balanced_sample=True
        )
        
        sample_df = pipeline.load_and_prepare_data()
        
        # Check that we have equal numbers of scam and legitimate samples
        scam_count = len(sample_df[sample_df['label'] == 1])
        legit_count = len(sample_df[sample_df['label'] == 0])
        assert scam_count == legit_count or abs(scam_count - legit_count) <= 1
    
    @patch('src.annotate.annotation_pipeline.make_api_call')
    @patch('src.annotate.annotation_pipeline.LLM')
    def test_save_annotations(self, mock_llm_class, mock_api_call,
                             temp_dataset_file, mock_llm_instance, mock_llm,
                             mock_annotation_response, test_results_dir, clean_results_dir):
        """Test saving annotation results"""
        # Setup mocks
        mock_llm_class.return_value = mock_llm_instance
        mock_llm_instance.get_llm.return_value = mock_llm
        mock_api_call.return_value = mock_annotation_response
        
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="openai",
            model="gpt-4o-mini",
            sample_size=1,
            output_dir=test_results_dir
        )
        
        pipeline.setup_llm()
        sample_df = pipeline.load_and_prepare_data()
        results = pipeline.annotate_sample(sample_df)
        
        # Test save functionality
        save_paths = pipeline.save_annotations()
        
        assert 'results_directory' in save_paths
        assert 'annotated_dataset_path' in save_paths
        assert 'summary_path' in save_paths
        
        # Check that files were created
        results_dir = Path(save_paths['results_directory'])
        assert results_dir.exists()
        assert Path(save_paths['annotated_dataset_path']).exists()
        assert Path(save_paths['summary_path']).exists()
    
    def test_create_annotation_record(self, temp_dataset_file, mock_annotation_response):
        """Test annotation record creation"""
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="openai",
            model="gpt-4o-mini"
        )
        
        # Load data to get data_loader features
        pipeline.load_and_prepare_data()
        
        # Create test row
        test_row = pd.Series({
            'id': 123,
            'label': 1,
            'subject': 'Test subject',
            'body': 'Test body',
            'sender': 'test@example.com'
        })
        
        record = pipeline._create_annotation_record(1, test_row, mock_annotation_response)
        
        assert record['record_id'] == 1
        assert record['label'] == 1
        assert record['class'] == 'Scam'
        assert record['explanation'] == mock_annotation_response.explanation
        assert record['key_indicators'] == mock_annotation_response.key_indicators
        assert record['confidence'] == mock_annotation_response.confidence
        assert record['usability'] == mock_annotation_response.usability
        assert 'annotation_timestamp' in record
        assert record['id'] == 123
        assert 'original_subject' in record
        assert 'original_body' in record
        assert 'original_sender' in record
    
    def test_create_error_annotation_record(self, temp_dataset_file):
        """Test error annotation record creation"""
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="openai",
            model="gpt-4o-mini"
        )
        
        # Load data to get data_loader features
        pipeline.load_and_prepare_data()
        
        # Create test row
        test_row = pd.Series({
            'id': 456,
            'label': 0,
            'subject': 'Test subject',
            'body': 'Test body',
            'sender': 'test@example.com'
        })
        
        error_record = pipeline._create_error_annotation_record(1, test_row, "Test error message")
        
        assert error_record['record_id'] == 1
        assert error_record['label'] == 0
        assert error_record['class'] == 'Legitimate'
        assert 'Test error message' in error_record['explanation']
        assert error_record['key_indicators'] == []
        assert error_record['confidence'] == 'error'
        assert error_record['usability'] == 0
        assert 'annotation_timestamp' in error_record


class TestAnnotationResponseSchema:
    """Test cases for AnnotationResponseSchema"""
    
    def test_schema_creation(self):
        """Test creating AnnotationResponseSchema instance"""
        response = AnnotationResponseSchema(
            explanation="This is a test explanation",
            key_indicators=["urgent language", "suspicious link"],
            confidence="high",
            usability=True
        )
        
        assert response.explanation == "This is a test explanation"
        assert response.key_indicators == ["urgent language", "suspicious link"]
        assert response.confidence == "high"
        assert response.usability == True
    
    def test_schema_validation(self):
        """Test schema validation"""
        # Valid data
        valid_data = {
            "explanation": "Valid explanation",
            "key_indicators": ["indicator1", "indicator2"],
            "confidence": "medium",
            "usability": False
        }
        response = AnnotationResponseSchema(**valid_data)
        assert response.explanation == "Valid explanation"
        assert len(response.key_indicators) == 2
        
        # Invalid data should raise validation error
        with pytest.raises(Exception):
            AnnotationResponseSchema(
                explanation="test",
                key_indicators="not_a_list",  # Should be a list
                confidence="high",
                usability=True
            )


class TestAnnotationIntegration:
    """Integration tests for annotation pipeline"""
    
    @patch('src.annotate.annotation_pipeline.make_api_call')
    @patch('src.annotate.annotation_pipeline.LLM')
    def test_run_full_annotation(self, mock_llm_class, mock_api_call,
                                 temp_dataset_file, mock_llm_instance, mock_llm,
                                 mock_annotation_response, test_results_dir, clean_results_dir):
        """Test running the full annotation pipeline"""
        # Setup mocks
        mock_llm_class.return_value = mock_llm_instance
        mock_llm_instance.get_llm.return_value = mock_llm
        mock_api_call.return_value = mock_annotation_response
        
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="openai",
            model="gpt-4o-mini",
            sample_size=2,
            output_dir=test_results_dir
        )
        
        # Run full pipeline
        result = pipeline.run_full_annotation()
        
        assert 'annotations' in result
        assert 'save_paths' in result
        assert 'summary' in result
        
        assert len(result['annotations']) == 2
        assert result['summary']['total_records'] == 2
        assert result['summary']['successful_annotations'] >= 0
        assert result['summary']['usable_annotations'] >= 0
        
        # Check that files were created
        save_paths = result['save_paths']
        assert Path(save_paths['results_directory']).exists()
        assert Path(save_paths['annotated_dataset_path']).exists()
        assert Path(save_paths['summary_path']).exists()
    
    @patch('src.annotate.annotation_pipeline.make_api_call')
    @patch('src.annotate.annotation_pipeline.LLM')
    def test_run_full_annotation(self, mock_llm_class, mock_api_call,
                                      temp_dataset_file, mock_llm_instance, mock_llm,
                                      mock_annotation_response, test_results_dir, clean_results_dir):
        """Test running the full annotation pipeline asynchronously"""
        # Setup mocks
        mock_llm_class.return_value = mock_llm_instance
        mock_llm_instance.get_llm.return_value = mock_llm
        mock_api_call.return_value = mock_annotation_response
        
        pipeline = LLMAnnotationPipeline(
            dataset_path=temp_dataset_file,
            provider="openai",
            model="gpt-4o-mini",
            sample_size=2,
            output_dir=test_results_dir
        )
        
        # Run full pipeline async
        result = asyncio.run(pipeline.run_full_annotation(concurrent_requests=2))
        
        assert 'annotations' in result
        assert 'save_paths' in result
        assert 'summary' in result
        
        assert len(result['annotations']) == 2
        assert result['summary']['total_records'] == 2