import ast
import os
import subprocess
import tempfile
from typing import Optional, Tuple

try:  # Support both package and path-based execution
    from .runtime_validation import perform_runtime_validation  # type: ignore
except ImportError:  # pragma: no cover
    from runtime_validation import perform_runtime_validation  # type: ignore


def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Python syntax using AST"""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"


def validate_code(file_path: str) -> Tuple[bool, Optional[str]]:
    """Validate code with syntax and basic runtime checks"""
    # First check syntax
    try:
        code = open(file_path, "r", encoding="utf-8").read()
        is_valid, error = validate_syntax(code)
        if not is_valid:
            return False, error
    except UnicodeDecodeError:
        try:
            code = open(file_path, "r", encoding="latin-1").read()
            is_valid, error = validate_syntax(code)
            if not is_valid:
                return False, error
        except Exception as exc:
            return False, f"File read error: {exc}"
    except Exception as exc:
        return False, f"File read error: {exc}"

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

    runtime_ok, runtime_error = perform_runtime_validation(file_path)
    if not runtime_ok:
        return False, runtime_error

    return True, None

validate_code("/Users/sandeephirani/Pictures/Inputs/input/main.py")
