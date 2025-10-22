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

    # Basic import test (safer than full execution)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(
                f"""
try:
    import sys
    sys.path.insert(0, '{os.path.dirname(file_path)}')

    # Try to parse and validate imports
    with open('{file_path}', 'r') as f:
        code = f.read()

    # Extract and test imports
    import ast
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                try:
                    __import__(alias.name)
                except ImportError:
                    pass  # Some imports might not be available in test env
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


def validate_repository(root_path: str) -> Tuple[bool, Optional[str]]:
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

    runtime_ok, runtime_error = perform_project_runtime_validation(root_path)
    if not runtime_ok:
        return False, runtime_error

    return True, None
