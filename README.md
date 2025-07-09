# Scam Detection Evaluation Pipeline

## Quick Setup Guide

### Step 1: Prerequisites

- Python 3.10+ (3.12 recommended)
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

### Step 5: Use Sample Datasets (Optional)

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

### Step 6: Prepare Your own Datasets
#### Place your dataset in the `data/cleaned` directory.

#### For the dataset you are working with:

Must be in CSV format with:
- A `label` column (1=scam, 0=legitimate)
- At least one content column


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
### Here is the dataset I used for evaluation:

**Email Phishing Evaluation:**
- Path: `data/cleaned/unified_phishing_email_dataset.csv`
- Required columns: `subject`, `body`, `label` (1=scam, 0=legitimate)

**SMS Phishing Evaluation:**
- Path: `data/cleaned/phishing_sms_dataset.csv`
- Required columns: `message`, `label` (1=scam, 0=legitimate)

**Error Dataset Evaluation:**
- The error dataset is complied from the datapoints that the LLM failed to classify.
- Path: `data/cleaned/unified_error_dataset/unified_error_dataset.csv`
- Required columns: `content`, `label` (1=scam, 0=legitimate)

### Step 7: Running Evaluations in small sample size

To run evaluations in a small sample size, use the provided evaluation scripts:

```python
# For email phishing evaluation
uv run python eval_email.py

# For SMS phishing evaluation  
uv run python eval_sms.py

# For error dataset evaluation
uv run python eval_error.py

# For generated transcript evaluation
uv run python transcript_eval.py
``` 


### Step 8: Customizing Evaluation Parameters

To modify evaluation parameters, edit the respective evaluation scripts:

**For Email Evaluation (`email_eval.py`):**
```python
# Modify these parameters in the email_eval() function
dataset_path = "data/cleaned/unified_phishing_email_dataset.csv"
provider = "openai"  # or "lm-studio", "anthropic", "gemini"
model = "gpt-4.1"
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
model = "gpt-4.1"
sample_size = 100
content_columns = ['content']
```

**For Evaluation over Generated Transcripts (`transcript_eval.py`):**
```python
# Modify these parameters in the transcript_eval() function
dataset_path = "results/generated_transcripts/{timestamp}/detailed_results.csv"
provider = "openai"  # or "lm-studio", "anthropic", "gemini"
model = "gpt-4.1"
sample_size = 100
content_columns = ['transcript']
```


## Transcript Generation Pipeline

The transcript generation pipeline creates realistic phone conversation transcripts for scam detection training. It uses alternating models to generate diverse, realistic conversations across different categories.

### Model Alternation Strategy

The pipeline uses two models with different specializations:

**Model A (Authority Scams, Tech Support, Legitimate Customer Service):**
- Authority impersonation scams (30%)
- Tech support scams (25%)
- Legitimate customer service calls (35%)
- Subtle scams (10%)

**Model B (Urgency Scams, Financial Fraud, Legitimate Personal/Business, Borderline):**
- Urgency-based scams (25%)
- Financial fraud calls (20%)
- Legitimate personal/business calls (30%)
- Borderline suspicious calls (15%)
- Subtle scams (10%)

### Usage

**Command Line Interface:**
```bash
# Generate transcript dataset with checkpoints
uv run python main.py
# Choose option 3 for transcript generation

# Or use the direct script
uv run python src/transcript_generator.py --sample-size 1000 --checkpoint-interval 100
```

**Test the functionality:**
```bash
# Run a small test to verify everything works
uv run python test_transcript_generation.py
```

### Output Format

Generated transcripts are saved in CSV format with the following columns:
- `transcript`: Complete phone conversation with speaker identification
- `classification`: LEGITIMATE, OBVIOUS_SCAM, BORDERLINE_SUSPICIOUS, or SUBTLE_SCAM
- `generation_model`: Which model generated the transcript
- `category_assigned`: Specific category of the conversation
- `conversation_length`: Approximate length in words
- `participant_demographics`: Brief description of participants
- `timestamp`: Generation timestamp

## Full Dataset Processing with Checkpointing

The existing pipeline classes (`LLMAnnotationPipeline`, `ScamDetectionEvaluator`, and `TranscriptGenerator`) now include built-in checkpointing capabilities, allowing you to process entire datasets without sampling limitations and resume from where you left off if interrupted.

#### Key Features:
- **Process entire datasets** without sampling limitations  
- **Configurable checkpointing** (save progress every N records)
- **Resume capability** to continue from where you left off
- **Both sync and async processing** for optimal performance
- **Support for both annotation and evaluation** tasks
- **Automatic error handling** and progress tracking
- **Integrated into existing pipeline classes** - no separate components needed

#### Quick Start Examples:

**1. Command Line Interface Usage:**

```bash
# Evaluate/Annotate entire dataset with checkpoints every X records
uv run python main.py
```

#### Checkpoint Management:

Checkpoints are automatically saved as JSON files containing:
- Current processing position
- All results processed so far
- Configuration used
- Timestamp and progress information

**Checkpoint filename format:**
```
{dataset_name}_{task_type}_{provider}_{model}_{timestamp}.json
```

**Example checkpoint structure:**
```
checkpoints/
├── unified_phishing_email_dataset_evaluation_lm-studio_unsloth-qwen3-30b-a3b_20240627_143022.json
├── unified_phishing_email_dataset_annotation_lm-studio_unsloth-qwen3-30b-a3b_20240627_150815.json
├── transcript_generation_openai_gpt-4_20240627_160000.json
└── ...
```

#### Error Handling:

- Individual record errors are logged and saved as error records
- Processing continues even if some records fail
- All progress is preserved in checkpoints
- Final results include success rates and error summaries

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
├── generated_transcripts/
│   └── {timestamp}/
│       ├── detailed_results.csv      # Generated transcripts with metadata
│       ├── generation_summary.json   # Generation statistics
│       └── summary_report.txt        # Human-readable generation report
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
