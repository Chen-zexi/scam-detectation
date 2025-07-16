# ScamAI Detection Pipeline Tests

This directory contains comprehensive test suites for the three main functionalities of the ScamAI detection pipeline:

1. **Evaluation Pipeline** (`test_evaluation.py`)
2. **Annotation Pipeline** (`test_annotation.py`) 
3. **Transcript Generation** (`test_transcript_generation.py`)

## Quick Start

### Install Test Dependencies

```bash
# Install test dependencies
pip install -r test/requirements-test.txt

# Or use the test runner to install dependencies
python test/test_runner.py --install-deps
```

### Run All Tests

```bash
# Run all tests
pytest test/

# Or use the test runner
python test/test_runner.py
```

### Run Specific Test Modules

```bash
# Run only evaluation tests
pytest test/test_evaluation.py
python test/test_runner.py evaluation

# Run only annotation tests  
pytest test/test_annotation.py
python test/test_runner.py annotation

# Run only transcript generation tests
pytest test/test_transcript_generation.py
python test/test_runner.py transcript
```

### Run with Coverage

```bash
# Generate coverage report
pytest test/ --cov=src --cov-report=html --cov-report=term-missing

# Or use the test runner
python test/test_runner.py --coverage
```

## Test Structure

### Test Files

- **`conftest.py`**: Shared fixtures and configuration
- **`test_evaluation.py`**: Tests for the evaluation pipeline
- **`test_annotation.py`**: Tests for the annotation pipeline  
- **`test_transcript_generation.py`**: Tests for transcript generation
- **`test_runner.py`**: Custom test runner with additional features
- **`requirements-test.txt`**: Test-specific dependencies

### Configuration Files

- **`pytest.ini`**: Pytest configuration (in project root)
- **`README.md`**: This documentation file

## Test Categories

### Unit Tests
- Test individual functions and methods
- Mock external dependencies (LLMs, API calls)
- Fast execution
- High coverage of edge cases

### Integration Tests  
- Test component interactions
- Test full pipeline workflows
- Mock only external API calls
- Validate data flow between components

### Schema Tests
- Validate Pydantic response schemas
- Test schema creation and validation
- Ensure backward compatibility

## Key Features

### Comprehensive Mocking
- **LLM Instances**: Mock LLM providers and models
- **API Calls**: Mock `make_api_call` and `make_api_call_async`
- **File Operations**: Use temporary files and directories
- **Environment Variables**: Set test environment variables

### Async Testing
- Full support for async/await patterns
- Test concurrent operations
- Validate async error handling

### Data Fixtures
- Sample datasets with realistic scam/legitimate examples
- Mock response objects for all schemas
- Temporary file management

### Error Testing
- Test API failures and timeouts
- Test malformed data handling
- Test resource constraints

## Test Coverage Areas

### Evaluation Pipeline (`test_evaluation.py`)
- ✅ Evaluator initialization and configuration
- ✅ LLM setup with structure models
- ✅ Data loading and sampling (balanced/unbalanced)
- ✅ Synchronous and asynchronous evaluation
- ✅ Structure model fallback mechanisms
- ✅ Thinking token handling
- ✅ Error handling and recovery
- ✅ Result saving and file management
- ✅ Performance metrics calculation
- ✅ EvaluationResponseSchema validation

### Annotation Pipeline (`test_annotation.py`)
- ✅ Pipeline initialization and configuration
- ✅ Prompt generation for different content types
- ✅ LLM setup and structure model configuration
- ✅ Synchronous and asynchronous annotation
- ✅ Generic API call integration
- ✅ Error handling and error record creation
- ✅ Annotation result saving and summarization
- ✅ AnnotationResponseSchema validation
- ✅ Full pipeline workflow testing

### Transcript Generation (`test_transcript_generation.py`)
- ✅ Generator initialization with dual models
- ✅ Model configuration and setup
- ✅ Category distribution calculation
- ✅ Prompt generation for different scam types
- ✅ Single transcript generation
- ✅ Batch generation with concurrency control
- ✅ Error handling and partial failure recovery
- ✅ Result saving and file management
- ✅ TranscriptResponseSchema validation
- ✅ Model selection filtering (A, B, both)

## Running Specific Test Types

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Run only async tests
pytest -m async

# Run only API-related tests
pytest -m api
```

### Run Tests with Specific Options

```bash
# Run with verbose output
pytest -v test/

# Run and stop on first failure
pytest -x test/

# Run in parallel (requires pytest-xdist)
pytest -n auto test/

# Run with timeout (requires pytest-timeout)
pytest --timeout=60 test/
```

## Mock Strategy

### LLM Mocking
All tests mock LLM interactions to:
- Avoid real API calls and costs
- Ensure deterministic test results
- Test error conditions safely
- Enable fast test execution

### API Call Mocking
The generic `make_api_call` and `make_api_call_async` functions are mocked to:
- Return predefined response schemas
- Test different response types
- Simulate API failures
- Validate parameter passing

### File System Mocking
Temporary files and directories are used to:
- Test file I/O operations safely
- Avoid polluting the real file system
- Enable parallel test execution
- Clean up automatically after tests

## Expected Test Output

### Successful Run
```
================================ test session starts ================================
platform darwin -- Python 3.11.0
collected 45 items

test/test_evaluation.py::TestScamDetectionEvaluator::test_init PASSED          [ 2%]
test/test_evaluation.py::TestScamDetectionEvaluator::test_setup_llm PASSED     [ 4%]
...
test/test_transcript_generation.py::TestTranscriptIntegration::test_run_full_generation PASSED [100%]

================================ 45 passed in 12.34s ================================
```

### With Coverage Report
```
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src/evaluate/evaluator.py                 234     12    95%   45-47, 123-125
src/annotate/annotation_pipeline.py       189      8    96%   67-69, 234-236
src/synthesize/transcript_generator.py    156      7    95%   89-91, 178-180
src/llm_core/api_call.py                   98      5    95%   23-25, 67-69
---------------------------------------------------------------------
TOTAL                                      677     32    95%
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running tests from the project root directory
2. **Missing Dependencies**: Install test requirements with `pip install -r test/requirements-test.txt`
3. **Async Test Failures**: Ensure `pytest-asyncio` is installed
4. **Coverage Issues**: Make sure the `src/` directory is in your Python path

### Environment Variables
Tests automatically set mock environment variables:
- `OPENAI_API_KEY=test-key-123`
- `ANTHROPIC_API_KEY=test-key-456`
- `GEMINI_API_KEY=test-key-789`
- `HOST_IP=localhost`

### Debug Mode
For debugging failed tests:

```bash
# Run with Python debugger
pytest --pdb test/

# Run with additional logging
pytest --log-cli-level=DEBUG test/

# Run specific test method
pytest test/test_evaluation.py::TestScamDetectionEvaluator::test_init -v
```

## Contributing

When adding new functionality:

1. **Add corresponding tests** in the appropriate test file
2. **Use existing fixtures** from `conftest.py` when possible
3. **Mock external dependencies** appropriately
4. **Add test markers** for categorization
5. **Update this README** if adding new test categories

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Use descriptive names that explain what is being tested

### Example Test Structure
```python
class TestMyFeature:
    """Test cases for MyFeature class"""
    
    def test_init(self):
        """Test feature initialization"""
        # Test basic initialization
        pass
    
    def test_method_with_valid_input(self):
        """Test method behavior with valid input"""
        # Test normal operation
        pass
    
    def test_method_with_invalid_input(self):
        """Test method behavior with invalid input"""
        # Test error handling
        pass
    
    @patch('module.external_dependency')
    def test_method_with_mocked_dependency(self, mock_dep):
        """Test method with mocked external dependency"""
        # Test with mocked dependencies
        pass
```

This test suite ensures the reliability and correctness of the ScamAI detection pipeline across all major functionalities.