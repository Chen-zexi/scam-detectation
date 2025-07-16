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

### Task 3: Transcript Generation

-   **What it does**: Synthesizes a new dataset of realistic phone call transcripts covering a wide range of scam and legitimate scenarios. This is useful for creating novel training data when real-world data is scarce.
-   **How to run**:
    1.  Run `python main.py`.
    2.  Choose option `3. Transcript Generation`.
    3.  Specify the number of transcripts to generate and select the LLM provider and model.
-   **Output**: A CSV file containing the generated transcripts, their classification (`OBVIOUS_SCAM`, `SUBTLE_SCAM`, `LEGITIMATE`, etc.), and other metadata.

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

### Transcript Generation Output

```
results/
└── generation/
    └── {timestamp}/
        ├── detailed_results.csv      # The generated transcripts and their metadata.
        ├── generation_summary.json   # Statistics about the generation process.
        └── summary_report.txt        # A human-readable summary.
```

## Codebase Structure

```
.
├── main.py                         # The main interactive entry point for all tasks.
└── src/
    ├── evaluate/                   # Evaluation task.
    │   └── evaluator.py
    ├── annotate/                   # Annotation task.
    │   └── annotation_pipeline.py
    ├── synthesize/                 # Synthetic data generation task.
    │   ├── transcript_generator.py
    │   └── transcript_prompts.py
    ├── llm_core/                   # A centralized module for handling all LLM interactions.
    │   ├── api_provider.py            # Manages connections to different LLM services.
    │   └── api_call.py                # Executes API requests.
    └── utils/                      # Helper modules for common tasks.
        ├── data_loader.py             # Loads and validates datasets.
        ├── metrics_calculator.py      # Computes performance metrics.
        └── results_saver.py           # Saves all outputs to the correct directory structure.
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