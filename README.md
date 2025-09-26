## Quick Start

### 1. Installation
```bash
git clone https://github.com/HarshithaSivalingala/cross_versioning.git
cd ml-upgrader

# Create virtual environment
python -m venv venv
source venv/bin/activate  
# On Windows:
 venv\Scripts\activate

# Install dependencies  
pip install -r requirements.txt
```

### 2. Get OpenRouter API Key
1. Sign up at [OpenRouter.ai](https://openrouter.ai/)
2. Get your API key from the dashboard
3. Add credits to your account (pay-per-use)

### 3. Set API Key
```bash
# Linux/Mac
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Windows
setx OPENROUTER_API_KEY "your-openrouter-api-key"
```

### 4. Run Web Interface
```bash
streamlit run ui/app.py
```

## Project Structure

```
ml-upgrader/
├── src/
│   ├── __init__.py
│   ├── utils.py               # Helper functions, diff generation
│   ├── llm_interface.py       # OpenRouter API integration
│   ├── validator.py           # Code validation (syntax + runtime)
│   ├── dependency_updater.py  # ML library version updates
│   ├── report_generator.py    # Markdown report generation
│   ├── agentic_upgrader.py    # Single file upgrade with retry logic
│   ├── repo_upgrader.py       # Repository-level orchestration  
│   └── cli.py                # Command line interface
├── ui/
│   └── app.py                # Streamlit web interface
├── tests/                    # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

