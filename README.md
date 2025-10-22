## Quick Start

### 1. Installation
```bash
git clone https://github.com/HarshithaSivalingala/cross_versioning.git

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

### 5. (Optional) Enable Runtime Validation
- Provide the runtime command either by setting `ML_UPGRADER_RUNTIME_COMMAND` **or** by adding `ml_upgrader_runtime.json` at the repo root (or `.ml-upgrader/runtime.json`). The environment variable takes precedence when both are present.
- The upgrader will create `.ml_upgrader_venv` inside the upgraded repo, install dependencies, and run the command after each LLM edit.
- On failures, stdout/stderr logs are captured and fed back to the LLM for the next retry.
- `ml_upgrader_runtime.json` structure:
  ```json
  {
    "command": ["python", "main.py"],
    "timeout": 180,
    "setup_commands": [
      "wget https://example.com/data.zip",
      ["python", "scripts/setup.py"]
    ],
    "skip_install": false,
    "force_reinstall": false,
    "shell": false,
    "cwd": "tests/unit",
    "env": {"PYTHONPATH": "src"},
    "max_log_chars": 6000
  }
  ```
  Fields are optional unless noted; remove any values you don't need. `command` accepts either a string (run via the shell) or a list of arguments. `setup_commands` runs before the main command and accepts shell strings or argument lists for more controlled execution.
- Optional knobs (env vars override config):
  - `ML_UPGRADER_RUNTIME_TIMEOUT` (seconds, default `120`)
  - `ML_UPGRADER_RUNTIME_SKIP_INSTALL=1` to skip dependency installs (useful for air-gapped runs)
  - `ML_UPGRADER_FORCE_REINSTALL=1` to force a clean reinstall of dependencies
  - `ML_UPGRADER_MAX_RUNTIME_LOG_CHARS` to control log truncation length
  - `ML_UPGRADER_RUNTIME_CONFIG` to point at a custom config path
- macOS archive artifacts (`__MACOSX` folders and `._filename` resource forks) and binary `.py` placeholders are automatically skipped during upgrades.

### Sample Legacy Project

- The repository ships with `examples/sample_project` and a ready-to-upload
  archive at `examples/sample_project.zip`.
- This project covers TensorFlow 1.x, PyTorch, and NumPy deprecations, plus
  runtime setup commands that generate a synthetic dataset before running a
  smoke test.
- Use it to exercise the full upgrade workflow: upload the zip (or point the
  CLI at the directory), let the agent modernise the code base, then re-run the
  bundled setup and runtime commands to confirm the upgrade.

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
├── examples/
│   ├── example1/             # Minimal TensorFlow 1.x + NumPy sample
│   ├── example1_upgraded/    # Result after an example upgrade pass
│   └── sample_project/       # Comprehensive legacy project with setup commands
├── tests/                    # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```
