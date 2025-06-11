# Scam Detection

## Download dataset
   ```bash
   # Download phishing email datasets
   wget "https://zenodo.org/api/records/8339691/files-archive" -O "Phishing_Email_Curated_Datasets.zip"
   
   # Extract datasets
   unzip "Phishing_Email_Curated_Datasets.zip" -d "Phishing_Email_Curated_Datasets"
   ```

## Quick Setup with uv

### Prerequisites

- Python 3.10+ 
- [uv](https://github.com/astral-sh/uv) package manager

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.sh | iex"

# Alternative: Install via pip
pip install uv
```

### Project Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chen-zexi/scam-detectation.git
   cd Scam-detection
   ```

2. **Create virtual environment and install dependencies**
   ```bash
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

## API Configuration

Set up LLM provider API keys as environment variables in `.env` file:

```bash
cp .env.example .env
```