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

### Step 5: Download and Prepare Sample Datasets (Optional)

```bash
# Create necessary directories
mkdir -p data/cleaned data/raw

# Download phishing email datasets (optional - for testing)
cd data/raw
wget "https://zenodo.org/api/records/8339691/files-archive" -O "Phishing_Email_Curated_Datasets.zip"

# Extract datasets
unzip "Phishing_Email_Curated_Datasets.zip"
cd ../..
```

### Step 5.5: Process Sample Datasets

### Step 6: For the dataset you are evaluating:

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

## Dataset Setup and Placement

### Step 7: Prepare Your Datasets

Place your datasets in the appropriate directories based on the type of evaluation:

**For Email Phishing Evaluation:**
- Place your email dataset at: `data/cleaned/unified_phishing_email_dataset.csv`
- Required columns: `subject`, `body`, `label` (1=scam, 0=legitimate)

**For SMS Phishing Evaluation:**
- Place your SMS dataset at: `data/cleaned/phishing_sms_dataset.csv`
- Required columns: `message`, `label` (1=scam, 0=legitimate)

**For Error Dataset Evaluation:**
- The error dataset is complied from the datapoints that the LLM failed to classify.
- Place your error dataset at: `data/cleaned/unified_error_dataset/unified_error_dataset.csv`
- Required columns: `content`, `label` (1=scam, 0=legitimate)

### Step 8: Running Evaluations

Use the specific evaluation scripts for different dataset types:

```bash
# Email Phishing Evaluation
python email_eval.py

# SMS Phishing Evaluation
python sms_eval.py

# Error Dataset Evaluation
python error_eval.py
```

### Step 9: Customizing Evaluation Parameters

To modify evaluation parameters, edit the respective evaluation scripts:

**For Email Evaluation (`email_eval.py`):**
```python
# Modify these parameters in the email_eval() function
dataset_path = "data/cleaned/unified_phishing_email_dataset.csv"
provider = "openai"  # or "lm-studio", "anthropic", "gemini"
model = "gpt-4"
sample_size = 500
content_columns = ['subject', 'body']
```

**For SMS Evaluation (`sms_eval.py`):**
```python
# Modify these parameters in the sms_eval() function
dataset_path = "data/cleaned/phishing_sms_dataset.csv"
provider = "openai"  # or "lm-studio", "anthropic", "gemini"
model = "gpt-4.1"
sample_size = 100
content_columns = ['message']
```

**For Evaluation over Error Dataset (`error_eval.py`):**
```python
# Modify these parameters in the error_eval() function
dataset_path = "data/cleaned/unified_error_dataset/unified_error_dataset.csv"
provider = "openai"  # or "lm-studio", "anthropic", "gemini"
model = "gpt-4"
sample_size = 100
content_columns = ['content']
```



## Python Script Usage

### Using Evaluation Scripts

The easiest way to run evaluations is using the provided evaluation scripts:

```python
# For email phishing evaluation
run email_eval.py

# For SMS phishing evaluation  
run sms_eval.py

# For error dataset evaluation
run error_eval.py
```

### Custom Evaluation Script

Create your own evaluation script based on the examples:

```python
#!/usr/bin/env python3
import sys
sys.path.append('src')
from src import ScamDetectionEvaluator

def custom_eval():
    # Customize these parameters
    dataset_path = "data/cleaned/your_dataset.csv"
    provider = "openai"
    model = "gpt-4"
    sample_size = 100
    
    evaluator = ScamDetectionEvaluator(
        dataset_path=dataset_path,
        provider=provider,
        model=model,
        sample_size=sample_size,
        random_state=42,
        content_columns=['your_content_column'],  # Specify your content columns
        balanced_sample=False
    )
    
    # Run complete evaluation
    results = evaluator.run_full_evaluation()
    print("✓ Evaluation completed successfully!")

if __name__ == "__main__":
    custom_eval()
```

### Advanced Usage

```python
from src import ScamDetectionEvaluator

# Manual control over each step
evaluator = ScamDetectionEvaluator(
    dataset_path='data/cleaned/your_dataset.csv',
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
