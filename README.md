# Scam Detection Evaluation Pipeline

## Quick Setup Guide

### Step 1: Prerequisites

- Python 3.10+ 
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Step 2: Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.sh | iex"

# Alternative: Install via pip
pip install uv
```

### Step 3: Clone Repository and Setup Environment

```bash
# Clone the repository
git clone https://github.com/Chen-zexi/scam-detectation.git
cd Scam-detection

# Create virtual environment
uv venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Step 4: API Configuration

Set up LLM provider API keys:

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your API keys
# For OpenAI:
OPENAI_API_KEY=your-openai-api-key

# For Anthropic:
ANTHROPIC_API_KEY=your-anthropic-api-key

# For Google Gemini:
GEMINI_API_KEY=your-gemini-api-key

# For Local Models (LM Studio):
LOCAL_API_KEY=your-local-api-key
HOST_IP=ip-of-your-localhost
```

## Data Setup

### Step 5: Download Sample Dataset

```bash
# Download phishing email datasets (optional - for testing)
wget "https://zenodo.org/api/records/8339691/files-archive" -O "Phishing_Email_Curated_Datasets.zip"

# Extract datasets
unzip "Phishing_Email_Curated_Datasets.zip" -d "Phishing_Email_Curated_Datasets"
```

### Step 6: Process Your Dataset

Your dataset must be in CSV format with:
- A `label` column (1=scam, 0=legitimate)
- At least one content column

**Example dataset formats:**

**Email Dataset (example):**
```csv
subject,body,sender,label
"Urgent: Verify Account","Click here to verify...",suspicious@fake.com,1
"Weekly Newsletter","Hello team...",newsletter@company.com,0
```

**Text Message Dataset (example):**
```csv
message,phone,timestamp,label
"You've won $1000! Reply now",+1234567890,2024-01-01,1
"Reminder: Doctor appointment",+1987654321,2024-01-01,0
```

## Running Examples

### Step 7: Test Your Setup

```bash
# Test with dry run (no API calls) You should expect to see dry run working.
cd src
python pipeline.py --dataset ../unified_phishing_email_dataset.csv --provider openai --model gpt-4.1 --sample-size 2 --dry-run

# Run with a small sample size
python pipeline.py --dataset ../unified_phishing_email_dataset.csv --provider openai --model gpt-4.1 --sample-size 2

# Run example scripts
cd ..
python example_usage.py
```

### Step 8: Run Your Own Evaluation

```bash
# Basic evaluation with all content features using openai gpt-4.1 for 10 samples
cd src
python pipeline.py --dataset ../your_dataset.csv --provider openai --model gpt-4.1 --sample-size 10

# Specify which columns to use as content (eg. subject and body) using openai gpt-4.1 for 100 samples
python pipeline.py --dataset ../your_dataset.csv --provider openai --model gpt-4.1 --content-columns subject body --sample-size 100
```

## Command Line Usage Guide

### Basic Commands

```bash
# Minimum required parameters (run from src directory)
cd src
python pipeline.py --dataset ../DATA.csv --provider PROVIDER --model MODEL

# Full parameter example
python pipeline.py \
  --dataset ../your_dataset.csv \
  --provider openai \
  --model gpt-4 \
  --sample-size 100 \
  --content-columns subject body sender \
  --random-state 42
```

### Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--dataset` | ✅ | Path to CSV dataset | `--dataset data.csv` |
| `--provider` | ✅ | LLM provider | `--provider openai` |
| `--model` | ✅ | Model name | `--model gpt-4` |
| `--sample-size` | ❌ | Number of samples to evaluate (default: 100) | `--sample-size 50` |
| `--content-columns` | ❌ | Specific columns to use (default: all) | `--content-columns subject body` |
| `--random-state` | ❌ | Random seed (default: 42) | `--random-state 123` |
| `--dry-run` | ❌ | Test without API calls | `--dry-run` |



## Python Script Usage

### Basic Usage

```python
from src import ScamDetectionEvaluator

# Use all available content features
evaluator = ScamDetectionEvaluator(
    dataset_path='your_dataset.csv',
    provider='openai',
    model='gpt-4.1',
    sample_size=100,
    random_state=42
)

# Run complete evaluation
results = evaluator.run_full_evaluation()
```

### Advanced Usage - Specify Content Columns

```python
# Only use specific columns as content
evaluator = ScamDetectionEvaluator(
    dataset_path='your_dataset.csv',
    provider='openai',
    model='gpt-4.1',
    sample_size=100,
    content_columns=['body']  # eg. Only use body
)

results = evaluator.run_full_evaluation()
```

### Step-by-Step Execution

```python
# Manual control over each step
evaluator = ScamDetectionEvaluator(
    dataset_path='data.csv',
    provider='openai',
    model='gpt-4',
    sample_size=100,
    content_columns=['subject', 'body']
)

# Execute step by step
evaluator.setup_llm()
sample_df = evaluator.load_and_prepare_data()
results = evaluator.evaluate_sample(sample_df)
metrics = evaluator.calculate_metrics()
save_paths = evaluator.save_results()
```

## Output Structure

Results are automatically saved to:

```
results/
├── {dataset_name}/
│   └── {timestamp}/
│       ├── detailed_results.csv      # Individual predictions with original data
│       ├── evaluation_metrics.csv    # Summary metrics
│       ├── evaluation_info.json      # Complete evaluation metadata
│       └── summary_report.txt        # Human-readable report
```

### Output Files Explained

1. **`detailed_results.csv`**: Contains evaluation results for each record with:
   - Original dataset features (prefixed with `original_`)
   - Prediction results (`predicted_label`, `predicted_class`)
   - Actual labels (`actual_label`, `actual_class`)
   - Correctness indicator (`is_correct`)
   - LLM reasoning (`llm_reason`)

2. **`evaluation_metrics.csv`**: Summary metrics including:
   - Accuracy, precision, recall, F1-score
   - Confusion matrix components
   - Dataset information
   - Model configuration

3. **`evaluation_info.json`**: Complete metadata about the evaluation run

4. **`summary_report.txt`**: Human-readable summary report
