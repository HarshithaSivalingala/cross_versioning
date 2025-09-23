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

def build_prompt(code: str, error: Optional[str] = None) -> str:
    """Build prompt for LLM with context about common migrations"""
    base_prompt = """You are an expert ML code migration assistant.
Rewrite this Python code to work with the latest versions of TensorFlow, PyTorch, NumPy, and JAX.

IMPORTANT MIGRATION PATTERNS:
- TensorFlow 1.x → 2.x: Remove tf.Session, use eager execution, tf.compat.v1 if needed
- PyTorch: Update deprecated functions, use newer tensor operations
- NumPy: Replace deprecated functions (np.asscalar → .item(), etc.)
- JAX: Update to current API patterns

Preserve ALL functionality. Return ONLY the corrected code without explanations.
"""
    
    if error:
        return f"""{base_prompt}

The following upgraded code failed with error:
{error}

Please fix the issue and return the full corrected file.

Code:
{code}
"""
    else:
        return f"""{base_prompt}

Code to upgrade:
{code}
"""

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