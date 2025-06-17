# Performance Evaluation Pipeline

A comprehensive pipeline for evaluating token/second performance of Large Language Models (LLMs) across different providers including OpenAI, Anthropic, Gemini, and local models.

## Features

- **Multi-provider Support**: Test models from OpenAI, Anthropic, Gemini, LM Studio, and vLLM
- **Token Counting**: Accurate token estimation using tiktoken
- **Performance Metrics**: Detailed timing and throughput measurements
- **Flexible Testing**: Use your own datasets or synthetic prompts
- **Comprehensive Reporting**: Detailed performance reports with statistics
- **Result Export**: Save results to CSV and JSON formats

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Set up your environment variables for the providers you want to test:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
export HOST_IP="your-local-server-ip"  # For local models
```

### 3. Run a Quick Test

```bash
# Test specific models
python performance_eval.py --models "openai:gpt-3.5-turbo,anthropic:claude-3-haiku-20240307" --sample-size 10

# Or use a configuration file
python performance_eval.py --config-file sample_performance_config.json
```

## Usage Examples

### Command Line Usage

```bash
# Compare multiple models with synthetic data
python performance_eval.py --models "openai:gpt-4,openai:gpt-3.5-turbo" --sample-size 30

# Use your own dataset
python performance_eval.py --models "openai:gpt-4" --dataset your_dataset.csv --content-columns subject body

# Use configuration file for complex setups
python performance_eval.py --config-file models_config.json --sample-size 50

# Save results to specific directory
python performance_eval.py --models "openai:gpt-3.5-turbo" --output-dir my_results
```

### Programmatic Usage

```python
from src.performance_evaluator import PerformanceEvaluator

# Initialize evaluator
evaluator = PerformanceEvaluator(
    sample_size=30,
    random_state=42
)

# Define models to test
models = [
    {"provider": "openai", "model": "gpt-4"},
    {"provider": "anthropic", "model": "claude-3-sonnet-20240229"}
]

# Run evaluation
results = evaluator.evaluate_multiple_models(models)

# Print report
evaluator.print_performance_report(results)

# Save results
save_paths = evaluator.save_results(results)
```

## Configuration File Format

Create a JSON configuration file for complex model testing:

```json
{
  "models": [
    {
      "provider": "openai",
      "model": "gpt-4",
      "description": "OpenAI GPT-4"
    },
    {
      "provider": "anthropic",
      "model": "claude-3-sonnet-20240229",
      "description": "Anthropic Claude 3 Sonnet"
    },
    {
      "provider": "gemini",
      "model": "gemini-pro",
      "description": "Google Gemini Pro"
    }
  ],
  "evaluation_settings": {
    "sample_size": 30,
    "random_state": 42,
    "output_dir": "performance_results"
  }
}
```

## Supported Providers

### OpenAI
- gpt-4
- gpt-3.5-turbo
- gpt-4-turbo

### Anthropic
- claude-3-sonnet-20240229
- claude-3-haiku-20240307
- claude-3-opus-20240229

### Google Gemini
- gemini-pro
- gemini-2.5-flash-exp-0827

### Local Models
- **LM Studio**: Set `HOST_IP` environment variable
- **vLLM**: Set `HOST_IP` environment variable
- Any OpenAI-compatible API endpoint

## Performance Metrics

The pipeline measures the following metrics for each model:

- **Response Time**: Total time for API call completion
- **Input Tokens**: Estimated tokens in the input prompt
- **Output Tokens**: Estimated tokens in the response
- **Total Tokens**: Input + output tokens
- **Tokens Per Second**: Throughput metric (total tokens / response time)
- **Success Rate**: Percentage of successful API calls

## Output Files

Results are saved in the specified output directory with timestamps:

- `detailed_performance_TIMESTAMP.csv`: Individual request results
- `performance_summary_TIMESTAMP.csv`: Aggregated statistics per model
- `performance_config_TIMESTAMP.json`: Configuration and metadata

## Example Output

```
================================================================================
PERFORMANCE EVALUATION REPORT
================================================================================
Model                     Success%   Avg T/s    Min T/s    Max T/s    Avg Time   Requests  
----------------------------------------------------------------------------------------------------
openai_gpt-3.5-turbo     100.0%     45.2       38.1       52.8       2.85       30        
anthropic_claude-3-haiku  100.0%     38.7       32.4       45.1       3.12       30        
gemini_gemini-pro        96.7%      42.1       35.2       48.9       2.95       30        

================================================================================
🏆 Best performing model: openai_gpt-3.5-turbo
   Average tokens/second: 45.2
   Success rate: 100.0%
```

## Dataset Format

If using your own dataset, it should be a CSV file with:
- A `label` column (required for the existing data loader)
- Content columns (text data to be used for prompts)

Example:
```csv
id,subject,body,label
1,"Meeting reminder","Don't forget about the meeting tomorrow",0
2,"Urgent: Verify account","Click here to verify your account immediately",1
```

## Advanced Usage

### Custom Prompt Types

The pipeline generates three types of synthetic prompts:
- **Short**: Simple questions (e.g., math problems)
- **Medium**: Analysis tasks with moderate context
- **Long**: Complex analysis with extensive context

### Token Estimation

Token counting uses the `tiktoken` library with model-specific encodings:
- GPT models: Official tiktoken encodings
- Other models: Fallback estimation based on text length

### Error Handling

The pipeline handles various error conditions:
- API failures
- Network timeouts
- Invalid responses
- Model initialization errors

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure environment variables are set correctly
2. **Import Errors**: Run `pip install -r requirements.txt`
3. **Local Model Connections**: Check HOST_IP and server status
4. **Rate Limiting**: Reduce sample size or add delays between requests

### Debug Mode

For debugging, you can catch and examine errors:

```python
try:
    results = evaluator.evaluate_multiple_models(models)
except Exception as e:
    print(f"Error: {e}")
    # Add specific error handling
```

## Contributing

To extend the pipeline:

1. Add new providers in `src/api_provider.py`
2. Implement custom metrics in `src/performance_evaluator.py`
3. Add new prompt types in the `_create_synthetic_prompts` method

## License

This project uses the same license as the parent scam detection project. 