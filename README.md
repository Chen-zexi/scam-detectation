# Scam Data Pipeline

## Quick Setup Guide

### 1. Prerequisites

- Python 3.10+ (3.12 recommended)
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/Chen-zexi/scam-detectation.git
cd scam-detectation

# Create and activate a virtual environment using uv
uv venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 3. API Configuration

Set up your LLM provider API keys.

```bash
# Copy the environment template
cp .env.example .env

# Edit the .env file and add your API keys and/or host IP
# For OpenAI, Anthropic, Gemini
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GEMINI_API_KEY=your-gemini-api-key

# For Local Models (LM Studio or vLLM)
HOST_IP=ip-of-your-local-machine
```

### 4. MongoDB Setup (Optional)

For enhanced synthesis features with knowledge base support:

```bash
# Start MongoDB using Docker (recommended)
docker-compose up -d

# Or install MongoDB locally
# macOS: brew install mongodb-community
# Ubuntu: sudo apt-get install mongodb
# Windows: Download from mongodb.com

# Add MongoDB configuration to .env
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=scam_detection
```

## Data Setup

For the **Evaluation** and **Annotation** tasks, you must provide your own dataset.

1.  **Create the directory**:
    ```bash
    mkdir -p data/cleaned
    ```
2.  **Place your dataset** inside the `data/cleaned/` directory. The framework will automatically discover any CSV files in this location.

### Dataset Format Requirements

-   **File Type**: Must be a CSV file.
-   **Label Column**: Must contain a `label` column where `1` indicates a scam and `0` indicates legitimate content.
-   **Content Columns**: Must contain at least one text-based content column (e.g., `body`, `message`, `transcript`).

**Example `data/cleaned/my_email_dataset.csv`:**
```csv
subject,body,sender,label
"Urgent: Verify Account","Click here to verify...",suspicious@fake.com,1
"Weekly Newsletter","Hello team...",newsletter@company.com,0
```

## Main Workflow: The Interactive CLI

The primary way to use the framework is through the `main.py` interactive script. It provides a guided, menu-driven experience for all major tasks.

To start, run:
```bash
# Using the original interface
uv run python main.py
```

The script will prompt you to select a task and configure it. Below is a detailed overview of each task.

---

### Task 1: Evaluation

-   **What it does**: Classifies records from your dataset using the selected LLM and compares the predictions to the ground-truth labels. This is useful for benchmarking model performance on scam detection.
-   **How to run**:
    1.  Run `python main.py`.
    2.  Choose option `2. Evaluation`.
    3.  Follow the prompts to select your dataset, LLM provider, model, and processing options.
-   **Output**: Generates a detailed report with accuracy, precision, recall, F1-score, and a confusion matrix.

---

### Task 2: Annotation

-   **What it does**: Uses an LLM to analyze records from your dataset and provide structured explanations for why the content is a scam or legitimate. This is ideal for enriching data or for educational purposes.
-   **How to run**:
    1.  Run `python main.py`.
    2.  Choose option `1. Annotation`.
    3.  Follow the prompts to select your dataset, LLM, and processing options.
-   **Output**: Creates a new, enriched dataset containing the LLM's explanations, key indicators of scam/legitimacy, confidence scores, and usability flags.

---

### Task 3: Synthesis

-   **What it does**: Generates synthetic datasets of various scam types including phone call transcripts, phishing emails, and SMS messages. Each type covers a wide range of scam and legitimate scenarios. This is useful for creating novel training data when real-world data is scarce.
-   **Available synthesis types**:
    - **Phone Transcripts**: Realistic phone conversations with tech support scams, authority impersonation, financial scams, etc.
    - **Phishing Emails**: Account verification, prize notifications, invoice scams, etc.
    - **SMS Scams**: Banking alerts, package delivery, verification codes, etc.
-   **How to run**:
    1.  Run `python main.py`.
    2.  Choose option `3. Synthesis` .
    3.  Select the synthesis type (phone, email, or SMS).
    4.  Choose a specific category or "ALL" for a mixed dataset.
    5.  Specify the number of items to generate and select the LLM provider and model.
-   **Output**: 
    - CSV file containing the generated content with appropriate fields for each type
    - Classifications (`OBVIOUS_SCAM`, `SUBTLE_SCAM`, `LEGITIMATE`, `PHISHING`, etc.)
    - Storage to MongoDB for enhanced data management

---

## Output Structure

All results are saved in a structured, timestamped format within the `results/` directory, making it easy to track and compare experiments.

### Evaluation Output

```
results/
└── {dataset_name}/
    └── {timestamp}/
        ├── detailed_results.csv      # Row-by-row predictions and original data.
        ├── evaluation_metrics.csv    # Key performance metrics in a single row.
        ├── evaluation_info.json      # Full configuration and metrics in JSON format.
        └── summary_report.txt        # Human-readable summary of the results.
```

### Annotation Output

```
results/
└── annotation/
    └── {dataset_name}/
        └── {timestamp}/
            ├── {dataset_name}_annotated.csv  # The new dataset with LLM explanations.
            └── annotation_summary.json         # Statistics about the annotation run.
```

### Synthesis Output

```
results/
└── synthesis/
    └── {synthesis_type}/           # e.g., phone_transcript, phishing_email, sms_scam
        └── {timestamp}/
            ├── synthesis_results.csv    # The generated content and metadata
            ├── synthesis_summary.json   # Statistics about the generation process
            └── synthesis_report.txt     # A human-readable summary
```

Note: If MongoDB is enabled, generated data is also stored in the database for easier querying and management.

## Codebase Structure

```
.
├── main.py                         # Interactive entry point
└── src/
    ├── evaluate/                   # Evaluation task
    │   ├── evaluator.py              # Main evaluation class
    │   ├── prompt_generator.py       # Dynamic prompt generation
    │   └── pipeline.py               # CLI pipeline for batch evaluation
    ├── annotate/                   # Annotation task
    │   └── annotation_pipeline.py    # LLM-based annotation generation
    ├── synthesize/                 # Synthetic data generation for all types
    │   ├── synthesis_generator.py    # Main synthesis class (handles all types)
    │   ├── synthesis_prompts.py      # Prompt management with DB support
    │   └── schema_builder.py         # Dynamic schema generation
    ├── llm_core/                   # Centralized LLM interactions
    │   ├── api_provider.py           # Unified interface for all LLM providers
    │   └── api_call.py              # Async/sync API call utilities
    ├── database/                   # MongoDB integration
    │   ├── mongodb_config.py         # Database configuration
    │   ├── knowledge_base_service.py # Knowledge base operations
    │   └── scam_data_service.py     # Synthetic data storage
    ├── cli/                        # CLI components
    │   ├── dataset_manager.py        # Dataset discovery and selection
    │   ├── model_selector.py         # Model selection interface
    │   ├── checkpoint_manager.py     # Checkpoint handling
    │   └── config_manager.py         # Configuration management
    ├── utils/                      # Helper modules
    │   ├── data_loader.py            # Dataset loading and validation
    │   ├── metrics_calculator.py     # Performance metrics
    │   └── results_saver.py          # Structured result persistence
    └── exceptions.py               # Custom exception hierarchy
```

## Advanced Usage

For non-interactive or scripted workflows, you can directly import and use the pipeline classes from the `src/` directory.

**Example: Scripted Evaluation**
```python
from src.evaluate import ScamDetectionEvaluator
import asyncio

async def run_evaluation():
    # Initialize the evaluator
    evaluator = ScamDetectionEvaluator(
        dataset_path="data/cleaned/your_dataset.csv",
        provider="openai",
        model="gpt-4.1-mini",
        sample_size=100,
        balanced_sample=True
    )

    # Run the evaluation asynchronously
    results = await evaluator.run_full_evaluation_async(concurrent_requests=10)
    print("Evaluation complete. Results saved to:", results['save_paths']['results_directory'])

if __name__ == "__main__":
    asyncio.run(run_evaluation())
```

**Example: Synthesis with MongoDB**
```python
from src.synthesize import SynthesisGenerator
import asyncio

async def generate_synthetic_data():
    # Initialize generator for phishing emails
    generator = SynthesisGenerator(
        synthesis_type="phishing_email",
        sample_size=50,
        category="ALL",  # Generate mixed categories
        provider="openai",
        model="gpt-4.1-mini",
        save_to_mongodb=True  # Enable MongoDB storage
    )
    
    # Generate with checkpointing support
    results = await generator.process_full_generation_with_checkpoints(
        checkpoint_interval=10,
        concurrent_requests=5
    )
    
    print(f"Generated {results['success_count']} items")
    print(f"Results saved to: {results['detailed_results']}")

if __name__ == "__main__":
    asyncio.run(generate_synthetic_data())
```