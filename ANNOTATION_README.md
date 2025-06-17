# LLM Annotation Pipeline for Scam Detection

This pipeline uses Large Language Models (LLMs) to generate detailed explanations for why content is classified as legitimate or scam. Unlike the evaluation pipeline that tests LLM performance, this annotation pipeline helps create training data by providing educational explanations for existing labeled data.

## What it does

The annotation pipeline:
1. Takes your labeled dataset (with correct scam/legitimate labels)
2. Shows each piece of content + its correct label to an LLM
3. Asks the LLM to explain WHY that classification is correct
4. Generates detailed educational annotations including:
   - Comprehensive explanations of the classification reasoning
   - Key indicators that support the classification
   - Confidence levels for the explanations

## Usage

### Quick Start

```bash
# Simple annotation with 20 samples
python annotate_data.py --dataset your_data.csv --provider openai --model gpt-4 --sample-size 20

# Balanced sampling (equal scam/legitimate examples)
python annotate_data.py --dataset your_data.csv --provider openai --model gpt-4 --sample-size 50 --balanced-sample

# Specify which columns to use as content
python annotate_data.py --dataset your_data.csv --provider openai --model gpt-4 --content-columns subject body
```

### Full Pipeline

```bash
# Run from src directory
cd src
python annotation_pipeline.py --dataset ../your_data.csv --provider openai --model gpt-4 --sample-size 100 --balanced-sample --output-dir ../annotations
```

## Parameters

### Required
- `--dataset`: Path to your CSV dataset (must have a 'label' column with 1=scam, 0=legitimate)
- `--provider`: LLM provider (`openai`, `anthropic`, `gemini`, `local`)
- `--model`: Model name (e.g., `gpt-4`, `claude-3-sonnet`, `gemini-pro`)

### Optional
- `--sample-size`: Number of records to annotate (default: 20 for quick script, 100 for full pipeline)
- `--balanced-sample`: Sample equal numbers of scam and legitimate examples
- `--content-columns`: Specific columns to use as content (e.g., `subject body`)
- `--output-dir`: Directory to save results (default: `annotations`)
- `--random-state`: Random seed for reproducibility (default: 42)

## Output

The pipeline generates:

1. **Detailed Annotations** (`*_annotations.csv`): CSV with all annotations including:
   - Original content
   - Correct labels
   - LLM explanations
   - Key indicators identified
   - Confidence levels
   - Timestamps

2. **Annotation Summary** (`annotation_summary.json`): Statistics including:
   - Success rates
   - Confidence distribution
   - Average indicators per class
   - Model and dataset info

## Example Output

For a scam email, the LLM might generate:

```
Explanation: "This is correctly classified as a SCAM because it exhibits multiple 
deceptive techniques: urgent language pressuring immediate action, impersonation 
of a legitimate bank, requests for sensitive credentials, and suspicious grammar 
patterns typical of phishing attempts."

Key Indicators: [
  "Urgent deadline pressure ('Act now or lose access')",
  "Requests for login credentials via email",
  "Suspicious sender domain not matching claimed bank",
  "Generic greeting instead of personalized message",
  "Grammatical errors and awkward phrasing"
]

Confidence: "high"
```

## Use Cases

1. **Training Data Creation**: Generate explanations for training other models
2. **Educational Content**: Create examples for security awareness training
3. **Feature Engineering**: Identify important indicators for detection models
4. **Data Validation**: Verify that your labels are correct and well-justified
5. **Research**: Understand what patterns LLMs identify in scam content

## Dataset Requirements

Your CSV dataset must include:
- A `label` column with 1 for scam, 0 for legitimate
- At least one content column (email body, message text, etc.)
- Any additional metadata columns you want preserved

Example dataset structure:
```csv
subject,body,sender,label
"Urgent: Account Suspended","Click here to verify...","noreply@bank-security.com",1
"Meeting Tomorrow","Hi John, let's meet at 2pm...","alice@company.com",0
```

## Notes

- The pipeline preserves all original data while adding annotations
- Error handling ensures partial results are saved even if some records fail
- Timestamps are added to track when annotations were generated
- The system is designed to work with any content type (emails, texts, conversations, etc.)

## Differences from Evaluation Pipeline

| Annotation Pipeline | Evaluation Pipeline |
|-------------------|-------------------|
| Uses correct labels as input | Tests LLM predictions |
| Generates explanations | Measures accuracy |
| Creates training data | Validates model performance |
| Educational focus | Performance focus | 