import os
import re
import shutil
import difflib
from typing import Dict, List, Optional

_UNNECESSARY_DIRECTORIES = {"__MACOSX", "__pycache__"}
_UNNECESSARY_FILES = {".DS_Store", "Thumbs.db"}

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
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def prune_directory(root: str) -> None:
    """Remove artifacts such as __MACOSX folders and resource forks."""
    for current_root, dirnames, filenames in os.walk(root):
        for dirname in list(dirnames):
            if dirname in _UNNECESSARY_DIRECTORIES or dirname.startswith("._"):
                shutil.rmtree(os.path.join(current_root, dirname), ignore_errors=True)
                dirnames.remove(dirname)
        for filename in list(filenames):
            if (
                filename in _UNNECESSARY_FILES
                or filename.startswith("._")
                or filename.endswith(".pyc")
            ):
                try:
                    os.remove(os.path.join(current_root, filename))
                except OSError:
                    continue


def is_probably_binary(path: str, sample_size: int = 2048) -> bool:
    """Heuristic to detect binary files (null bytes or low text ratio)."""
    try:
        with open(path, "rb") as fh:
            sample = fh.read(sample_size)
    except OSError:
        return False

    if not sample:
        return False

    if b"\x00" in sample:
        return True

    text_chars = bytes({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x7F)))
    text_set = set(text_chars)
    text_count = sum(1 for byte in sample if byte in text_set)
    ratio = text_count / len(sample)
    return ratio < 0.85


def should_skip_for_upgrade(path: str) -> Optional[str]:
    """Return reason string if file should not be upgraded."""
    parts = os.path.normpath(path).split(os.sep)
    if any(part == "__MACOSX" for part in parts):
        return "Skipped macOS resource fork metadata"

    filename = os.path.basename(path)
    if filename.startswith("._"):
        return "Skipped macOS resource fork file"

    if is_probably_binary(path):
        return "Skipped binary/non-text file"

    return None

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

def build_prompt_with_context(
    code: str, 
    dependency_context: Dict[str, str],
    error: Optional[str] = None
) -> str:
    """
    Build prompt for LLM with awareness of dependency interfaces.
    
    This helps the LLM maintain compatibility with files this code imports from.
    By showing the LLM the interfaces of dependencies, it can ensure the upgraded
    code calls functions correctly and uses the right types.
    
    Args:
        code: The code to upgrade
        dependency_context: Dict mapping file paths to their interface summaries
        error: Optional error from previous attempt
    
    Returns:
        Complete prompt string for the LLM
    """
    
    base_prompt = (
        "You are an expert Python ML code migration assistant.\n"
        "Upgrade the following Python code to be fully compatible with the latest stable version(s) "
        "of ONLY the libraries it already uses.\n\n"
        "⚠️  RULES:\n"
        "- Do NOT convert between frameworks (e.g., keep TensorFlow code in TensorFlow, PyTorch in PyTorch).\n"
        "- Do NOT add new frameworks unless already imported in the code.\n"
        "- Preserve all functionality and logic exactly.\n"
        "- Apply only necessary migrations (remove deprecated APIs, update function signatures, fix types).\n"
        "- Always return the ENTIRE corrected code.\n"
    )
    
    # Add dependency context if available
    if dependency_context:
        context_section = "\n\n📚 DEPENDENCY CONTEXT:\n"
        context_section += "This file imports from other files in the repository. "
        context_section += "Here are their current interfaces (already upgraded):\n\n"
        
        for dep_path, interface in dependency_context.items():
            dep_name = os.path.basename(dep_path)
            context_section += f"### {dep_name} ###\n"
            if interface.strip():
                context_section += interface + "\n\n"
            else:
                context_section += "(empty or no public interface)\n\n"
        
        context_section += (
            "⚠️  IMPORTANT: Your upgraded code MUST be compatible with these interfaces.\n"
            "- Match function signatures exactly\n"
            "- Use the same return types\n"
            "- Don't assume different APIs than shown above\n"
            "- If a function signature shows specific types, use them\n\n"
        )
        
        base_prompt += context_section
    
    base_prompt += (
        "```python\n"
        "# upgraded code here\n"
        "```"
    )
    
    # Add error context if this is a retry
    if error:
        return (
            f"{base_prompt}\n\n"
            "The previously upgraded code failed with this error:\n"
            f"{error}\n\n"
            "Please fix the issue and return the full corrected file.\n\n"
            f"Code to upgrade:\n{code}\n"
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
