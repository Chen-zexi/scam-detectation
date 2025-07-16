"""
Pytest configuration and shared fixtures for ScamAI tests
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
from pydantic import BaseModel
from typing import List

# Test schemas for mocking
class MockEvaluationResponse(BaseModel):
    Phishing: bool
    Reason: str

class MockAnnotationResponse(BaseModel):
    explanation: str
    key_indicators: List[str]
    confidence: str
    usability: bool

class MockTranscriptResponse(BaseModel):
    transcript: str
    classification: str
    category_assigned: str
    conversation_length: int
    participant_demographics: str
    timestamp: str

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing"""
    data = {
        'id': [1, 2, 3, 4, 5],
        'label': [1, 0, 1, 0, 1],
        'subject': [
            'Urgent: Your account will be suspended',
            'Welcome to our newsletter',
            'You won $1,000,000!',
            'Meeting reminder for tomorrow',
            'Click here to verify your account'
        ],
        'body': [
            'Your account will be suspended unless you verify immediately.',
            'Thank you for subscribing to our monthly updates.',
            'Congratulations! You have won our grand prize lottery.',
            'This is a reminder about tomorrows team meeting at 2 PM.',
            'Please click the link below to verify your account details.'
        ],
        'sender': [
            'noreply@suspicious-bank.com',
            'newsletter@legitimate-company.com',
            'winner@fake-lottery.org',
            'calendar@company.com',
            'security@phishing-site.net'
        ]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_dataset_file(sample_dataset):
    """Create a temporary CSV file with sample data"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataset.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def mock_llm():
    """Create a mock LLM instance"""
    mock = Mock()
    mock.invoke = Mock()
    mock.ainvoke = Mock()
    mock.with_structured_output = Mock()
    return mock

@pytest.fixture
def mock_llm_instance():
    """Create a mock LLM provider instance"""
    mock = Mock()
    mock.get_llm = Mock()
    mock.get_structure_model = Mock()
    return mock

@pytest.fixture
def mock_evaluation_response():
    """Mock evaluation response"""
    return MockEvaluationResponse(
        Phishing=True,
        Reason="Contains urgent language and suspicious sender"
    )

@pytest.fixture
def mock_annotation_response():
    """Mock annotation response"""
    return MockAnnotationResponse(
        explanation="This email uses urgent language to create pressure and contains suspicious sender address",
        key_indicators=["urgent language", "suspicious sender", "verification request"],
        confidence="high",
        usability=True
    )

@pytest.fixture
def mock_transcript_response():
    """Mock transcript response"""
    return MockTranscriptResponse(
        transcript="Caller: Hello, this is from your bank. We need to verify your account...",
        classification="OBVIOUS_SCAM",
        category_assigned="financial_scam",
        conversation_length=15,
        participant_demographics="elderly_victim_young_scammer",
        timestamp="2024-01-01T12:00:00"
    )

@pytest.fixture
def sample_output_dir():
    """Create a temporary output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def test_results_dir():
    """Create a temporary results directory and clean up after test"""
    import shutil
    
    # Create a unique test results directory
    test_dir = Path("test_results_temp")
    test_dir.mkdir(exist_ok=True)
    
    yield str(test_dir)
    
    # Clean up after test
    if test_dir.exists():
        shutil.rmtree(test_dir)

@pytest.fixture
def clean_results_dir():
    """Clean up any test artifacts in the main results directory after test"""
    import shutil
    import time
    
    # Store initial state
    results_dir = Path("results")
    initial_contents = set()
    if results_dir.exists():
        initial_contents = {item.name for item in results_dir.iterdir()}
    
    yield
    
    # Clean up test artifacts in results directory
    if results_dir.exists():
        current_contents = {item.name for item in results_dir.iterdir()}
        new_items = current_contents - initial_contents
        
        for item_name in new_items:
            item_path = results_dir / item_name
            try:
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                elif item_path.is_file():
                    item_path.unlink()
            except (PermissionError, FileNotFoundError):
                # Handle case where files might be locked or already deleted
                pass
        
        # Also remove any test-related subdirectories from before
        for item in results_dir.iterdir():
            if item.is_dir() and ("test" in item.name.lower() or "temp" in item.name.lower()):
                try:
                    shutil.rmtree(item)
                except (PermissionError, FileNotFoundError):
                    pass

@pytest.fixture(scope="session", autouse=True)
def cleanup_after_all_tests():
    """Clean up all test artifacts after all tests complete"""
    yield
    
    # Final cleanup after all tests
    import shutil
    
    # Clean up test results directory
    test_dir = Path("test_results_temp")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Clean up any remaining test artifacts in results directory
    results_dir = Path("results")
    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_dir() and ("test" in item.name.lower() or "temp" in item.name.lower() or
                                 item.name.startswith("generated_transcripts_") or
                                 item.name.startswith("annotated_") or 
                                 "evaluation_" in item.name.lower()):
                try:
                    shutil.rmtree(item)
                except (PermissionError, FileNotFoundError):
                    pass
    
    # Clean up any test checkpoints
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        for item in checkpoints_dir.iterdir():
            if item.is_file() and ("test" in item.name.lower() or "temp" in item.name.lower()):
                try:
                    item.unlink()
                except (PermissionError, FileNotFoundError):
                    pass

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Setup test environment variables"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-456")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-789")
    monkeypatch.setenv("HOST_IP", "localhost")