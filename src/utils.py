import os
import re
import difflib
from typing import Dict, List, Tuple, Optional

def read_file(path: str) -> str:
    """Read file content with encoding handling"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()

def write_file(path: str, content: str) -> None:
    """Write content to file with directory creation"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

from typing import Optional

def build_prompt_best(code: str, error: Optional[str] = None) -> str:
    """
    Build a robust, future-proof LLM prompt for ML code upgrade.
    
    Guarantees:
    - Detects which frameworks/libraries are used (TF, PyTorch, NumPy, JAX, etc.)
    - Upgrades only those frameworks to their latest stable versions
    - Preserves all logic and functionality
    - Removes deprecated APIs, outdated patterns, and placeholders in TensorFlow
    - Returns fully working code ready to run
    - Output ONLY inside a single fenced Python code block
    """
    
    base_prompt = (
        "You are an expert Python ML code migration assistant.\n"
        "Upgrade the following Python code to be fully compatible with the latest stable version(s) "
        "of ONLY the libraries it already uses.\n\n"
        "⚠️ RULES:\n"
        "- Do NOT convert frameworks (keep TensorFlow code as TensorFlow, PyTorch as PyTorch, etc.)\n"
        "- Detect and replace all deprecated APIs, functions, and patterns with the current recommended approach\n"
        "- Preserve all functionality and logic exactly; do not refactor unrelated parts\n"
        "- Ensure the code runs correctly with the latest stable release of each used library\n"
        "- Always return the ENTIRE upgraded code\n"
        "- Output ONLY a single fenced Python code block:\n\n"
        "```python\n"
        "# upgraded code here\n"
        "```"
    )
    
    if error:
        return (
            f"{base_prompt}\n\n"
            "The previously upgraded code failed with this error:\n"
            f"{error}\n\n"
            "Please fix it and return the full corrected file.\n\n"
            f"Code:\n{code}\n"
        )
    else:
        return (
            f"{base_prompt}\n\n"
            "Code to upgrade:\n"
            f"{code}\n"
        )


def build_prompt(code: str, error: Optional[str] = None) -> str:
    """Build prompt for LLM to upgrade ML/NumPy code in-place without mixing frameworks"""
    
    base_prompt = (
        "You are an expert Python ML code migration assistant.\n"
        "Upgrade the following Python code to be fully compatible with the latest stable version(s) "
        "of ONLY the libraries it already uses.\n\n"
        "⚠️ RULES:\n"
        "- Do NOT convert between frameworks (e.g., keep TensorFlow code in TensorFlow, PyTorch in PyTorch).\n"
        "- Do NOT add new frameworks unless already imported in the code.\n"
        "- Preserve all functionality and logic exactly.\n"
        "- Apply only necessary migrations (remove deprecated APIs, update function signatures, fix types).\n"
        "- Always return the ENTIRE corrected code.\n"
        "```python\n"
        "# upgraded code here\n"
        "```"
    )
    
    if error:
        return (
            f"{base_prompt}\n\n"
            "The previously upgraded code failed with this error:\n"
            f"{error}\n\n"
            "Please fix the issue and return the full corrected file.\n\n"
            f"Code:\n{code}\n"
        )
    else:
        return (
            f"{base_prompt}\n\n"
            "Code to upgrade:\n"
            f"{code}\n"
        )

def extract_api_changes(old_code: str, new_code: str) -> List[str]:
    """Extract API changes between old and new code"""
    changes = []
    
    # Common API patterns to detect
    patterns = {
        r'tf\.Session\(\)': 'Removed tf.Session (TF 1.x → 2.x)',
        r'tf\.placeholder': 'Replaced tf.placeholder with tf.Variable or function parameters',
        r'np\.asscalar': 'Replaced np.asscalar with .item()',
        r'torch\.cuda\.FloatTensor': 'Updated torch.cuda.FloatTensor to modern tensor creation',
        r'tf\.get_variable': 'Replaced tf.get_variable with tf.Variable',
        r'tf\.layers\.': 'Migrated tf.layers to tf.keras.layers',
        r'tf\.contrib\.': 'Removed tf.contrib (deprecated in TF 2.x)',
        r'np\.int\b': 'Replaced np.int with int',
        r'np\.float\b': 'Replaced np.float with float',
        r'torch\.autograd\.Variable': 'Removed torch.autograd.Variable (no longer needed)',
    }
    
    for pattern, description in patterns.items():
        if re.search(pattern, old_code) and not re.search(pattern, new_code):
            changes.append(description)
    
    return changes

def generate_diff(old_content: str, new_content: str, filename: str) -> str:
    """Generate unified diff between old and new content"""
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"old/{filename}",
        tofile=f"new/{filename}",
        n=3
    )
    return ''.join(diff)