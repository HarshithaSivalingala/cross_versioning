import ast
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from src.runtime_validation import perform_project_runtime_validation, perform_runtime_validation
from src import utils


def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Python syntax using AST"""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"


def validate_tf2_general(code: str) -> Tuple[bool, List[str]]:
    """General TF2 quality checks - catches issues across all patterns."""
    issues = []
    
    # Check 1: TF1-only APIs that must be removed
    tf1_only_apis = {
        'tf.Session': 'Remove tf.Session - use eager execution',
        'sess.run': 'Remove sess.run - use eager execution',
        'tf.placeholder': 'Remove tf.placeholder - use function parameters',
        'tf.get_variable': 'Replace tf.get_variable with tf.Variable',
        'tf.variable_scope': 'Replace tf.variable_scope with Python classes',
        'tf.contrib.': 'Remove tf.contrib.* - entire module removed in TF2',
        'slim.': 'Replace slim.* with tf.data.Dataset and tf.keras',
        'tf.app.run': 'Replace tf.app.run with standard Python main',
        'tf.flags': 'Replace tf.flags with argparse or absl.flags',
        'tf.logging': 'Replace tf.logging with Python logging module',
    }
    
    for api, fix in tf1_only_apis.items():
        if api in code:
            issues.append(fix)
    
    # Check 2: Python 2 compatibility (not needed)
    if 'from __future__ import' in code:
        issues.append("Remove 'from __future__ import' statements (Python 2 compatibility not needed in TF2)")
    
    # Check 3: File patterns without glob
    if 'TFRecordDataset' in code:
        # Check if there's a pattern variable being used
        has_pattern = any(x in code for x in ['%s', '%.format', 'f"', "f'", '.format('])
        has_glob = 'tf.io.gfile.glob' in code or 'tf.io.matching_files' in code
        
        if has_pattern and not has_glob:
            issues.append("Use tf.io.gfile.glob() to resolve file patterns before passing to TFRecordDataset")
    
    # Check 4: Missing performance optimizations
    if 'tf.data' in code and '.map(' in code:
        if 'num_parallel_calls' not in code and 'AUTOTUNE' not in code:
            issues.append("Add num_parallel_calls=tf.data.AUTOTUNE to .map() calls for performance")
    
    # Check 5: Dataset functions returning dicts
    if any(x in code for x in ['def get_split', 'def get_dataset', 'def load_data', 'def create_dataset']):
        # Check if function returns dict with metadata keys
        if "return {" in code:
            metadata_keys = ["'reader'", "'decoder'", "'data_sources'", '"reader"', '"decoder"', '"data_sources"']
            if any(key in code for key in metadata_keys):
                issues.append("Dataset functions must return tf.data.Dataset objects, not metadata dictionaries")
    
    # Check 6: VarLenFeature without sparse conversion
    if 'VarLenFeature' in code and 'parse_single_example' in code:
        if 'tf.sparse.to_dense' not in code and 'sparse.to_dense' not in code:
            issues.append("VarLenFeature returns sparse tensors - use tf.sparse.to_dense() to convert to dense")
    
    # Check 7: Awkward default values
    if 'default_value=tf.zeros(' in code or 'default_value=tf.constant(' in code:
        issues.append("Use simple default values (like -1 or '') instead of tf.zeros() or tf.constant()")
    
    # Check 8: Empty or useless operations
    if 'with_options(tf.data.Options())' in code:
        issues.append("Remove .with_options(tf.data.Options()) - it does nothing with empty options")
    
    # Check 9: Experimental APIs
    if 'tf.data.experimental' in code or 'tf.contrib.data' in code:
        issues.append("Replace experimental APIs with stable TF2 alternatives")
    
    # Check 10: Old-style feature specs
    if 'tf.FixedLenFeature' in code and 'tf.io.FixedLenFeature' not in code:
        # Check if they're using old style without tf.io prefix
        if 'import tensorflow as tf' in code:
            issues.append("Use tf.io.FixedLenFeature instead of tf.FixedLenFeature (API moved to tf.io)")
    
    return len(issues) == 0, issues


def _load_code(file_path: str, preloaded_code: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if preloaded_code is not None:
        return preloaded_code, None
    try:
        return utils.read_file(file_path), None
    except Exception as exc:
        return None, f"File read error: {exc}"


def validate_code(
    file_path: str,
    *,
    run_runtime: bool = True,
    preloaded_code: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate code with syntax and basic runtime checks"""
    code, load_error = _load_code(file_path, preloaded_code)
    if load_error:
        return False, load_error

    is_valid, error = validate_syntax(code or "")
    if not is_valid:
        return False, error

    # TF2-specific validation
    if "tensorflow" in (code or "") or "import tf" in (code or ""):
        is_valid_tf2, tf2_issues = validate_tf2_general(code or "")
        if not is_valid_tf2:
            return False, "TF2 quality issues:\n" + "\n".join(f"  - {issue}" for issue in tf2_issues)

    # Compile check
    try:
        subprocess.run(
            ["python", "-m", "py_compile", file_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        return False, f"Compilation error: {exc.stderr}"

    # Basic import test
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(
                f"""
try:
    import sys
    sys.path.insert(0, '{os.path.dirname(file_path)}')

    with open('{file_path}', 'r') as f:
        code = f.read()

    import ast
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                try:
                    __import__(alias.name)
                except ImportError:
                    pass
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                try:
                    __import__(node.module)
                except ImportError:
                    pass

    print("VALIDATION_SUCCESS")
except Exception as e:
    print(f"VALIDATION_ERROR: {{e}}")
"""
            )
            tmp.flush()

            result = subprocess.run(
                ["python", tmp.name],
                capture_output=True,
                text=True,
                timeout=10,
            )

            os.unlink(tmp.name)

            if "VALIDATION_ERROR" in result.stdout:
                error = result.stdout.split("VALIDATION_ERROR: ")[1].strip()
                return False, f"Import validation error: {error}"
            if "VALIDATION_SUCCESS" not in result.stdout and result.stderr:
                return False, f"Validation error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Validation timeout"
    except Exception as exc:
        return False, f"Validation error: {exc}"

    if run_runtime:
        runtime_ok, runtime_error = perform_runtime_validation(file_path)
        if not runtime_ok:
            return False, runtime_error

    return True, None


@dataclass
class CodeFile:
    path: str
    content: str


def validate_repository(
    root_path: str,
    *,
    runtime_output_dir: Optional[str] = None,
    runtime_compare_dir: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate all Python files under root_path, then run runtime validation once."""
    code_files: List[CodeFile] = []
    for current_root, _, filenames in os.walk(root_path):
        if "__pycache__" in current_root or current_root.endswith("__pycache__"):
            continue
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            file_path = os.path.join(current_root, filename)
            if utils.should_skip_for_upgrade(file_path):
                continue
            try:
                content = utils.read_file(file_path)
            except Exception as exc:
                return False, f"File read error: {exc}"
            code_files.append(CodeFile(file_path, content))

    for code_file in code_files:
        is_valid, error = validate_code(code_file.path, run_runtime=False, preloaded_code=code_file.content)
        if not is_valid:
            return False, error

    if not code_files:
        return True, None

    runtime_ok, runtime_error = perform_project_runtime_validation(
        root_path,
        output_capture_dir=runtime_output_dir,
        compare_with_dir=runtime_compare_dir,
    )
    if not runtime_ok:
        return False, runtime_error

    return True, None