import json
import os
import re
import shlex
import subprocess
import sys
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

RUNTIME_CONFIG_FILENAMES = (
    "ml_upgrader_runtime.json",
    os.path.join(".ml-upgrader", "runtime.json"),
)

_RUNTIME_SEARCH_SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
}


def perform_runtime_validation(file_path: str) -> Tuple[bool, Optional[str]]:
    project_root = _detect_project_root(file_path)
    return perform_project_runtime_validation(project_root)


def perform_project_runtime_validation(project_root: str) -> Tuple[bool, Optional[str]]:
    project_root = os.path.abspath(project_root)
    if not os.path.isdir(project_root):
        return False, f"Project root '{project_root}' not found"

    runtime_settings, runtime_error = _resolve_runtime_settings(project_root)
    if runtime_error:
        return False, runtime_error

    if runtime_settings:
        success, runtime_error = _run_runtime_validation(
            project_root,
            runtime_settings["command"],
            timeout=runtime_settings["timeout"],
            skip_install=runtime_settings["skip_install"],
            force_reinstall=runtime_settings["force_reinstall"],
            log_limit=runtime_settings["log_limit"],
            extra_env=runtime_settings["env"],
            runtime_cwd=runtime_settings["cwd"],
            shell_preference=runtime_settings["shell_preference"],
            command_label=runtime_settings["command_label"],
        )
        if not success:
            return False, runtime_error

    return True, None


def _detect_project_root(file_path: str) -> str:
    abs_path = os.path.abspath(file_path)

    env_root = os.getenv("ML_UPGRADER_PROJECT_ROOT")
    if env_root:
        env_root_abs = os.path.abspath(env_root)
        try:
            common = os.path.commonpath([abs_path, env_root_abs])
        except ValueError:
            common = None
        if common == env_root_abs:
            return env_root_abs

    current = os.path.dirname(abs_path)

    marker_names: List[str] = [
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
    ]
    marker_names.extend(RUNTIME_CONFIG_FILENAMES)

    while True:
        for marker in marker_names:
            candidate = os.path.join(current, marker)
            if os.path.exists(candidate):
                return current

        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    return os.path.dirname(abs_path)


def _select_venv_path(project_root: str) -> str:
    override = os.getenv("ML_UPGRADER_VENV_PATH")
    if override:
        return override

    default_candidates = [
        os.path.join(project_root, ".venv"),
        os.path.join(project_root, ".ml_upgrader_venv"),
    ]

    for candidate in default_candidates:
        _, python_path, _ = _resolve_venv_paths(candidate)
        if os.path.isfile(python_path):
            return candidate
        if os.path.isdir(candidate):
            return candidate

    return default_candidates[0]


def _max_runtime_log_chars(config_limit: Optional[int] = None) -> int:
    env_value = os.getenv("ML_UPGRADER_MAX_RUNTIME_LOG_CHARS")
    if env_value is not None:
        try:
            return max(0, int(env_value))
        except ValueError:
            pass

    if config_limit is not None:
        try:
            return max(0, int(config_limit))
        except (TypeError, ValueError):
            pass

    return 4000


def _runtime_timeout(config_timeout: Optional[int] = None) -> int:
    env_value = os.getenv("ML_UPGRADER_RUNTIME_TIMEOUT")
    if env_value is not None:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass

    if config_timeout is not None:
        try:
            return max(1, int(config_timeout))
        except (TypeError, ValueError):
            pass

    return 120


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _resolve_bool_option(env_name: str, config_value: Optional[Any], default: bool = False) -> bool:
    env_value = os.getenv(env_name)
    if env_value is not None:
        parsed = _parse_bool(env_value)
        if parsed is not None:
            return parsed

    if config_value is not None:
        parsed = _parse_bool(config_value)
        if parsed is not None:
            return parsed

    return default


def _load_runtime_config(project_root: str) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    explicit = os.getenv("ML_UPGRADER_RUNTIME_CONFIG")
    candidate_paths: List[str] = []

    if explicit:
        candidate_paths.append(explicit if os.path.isabs(explicit) else os.path.join(project_root, explicit))

    for name in RUNTIME_CONFIG_FILENAMES:
        candidate_paths.append(os.path.join(project_root, name))

    seen = set()
    for path in candidate_paths:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except json.JSONDecodeError as exc:
                return {}, path, f"Runtime config parse error in {path}: {exc}"
            except OSError as exc:
                return {}, path, f"Runtime config read error in {path}: {exc}"

            if not isinstance(data, dict):
                return {}, path, f"Runtime config {path} must be a JSON object"

            runtime_section = data.get("runtime")
            if runtime_section is not None:
                if not isinstance(runtime_section, dict):
                    return {}, path, f"Runtime config {path} field 'runtime' must be an object"
                return runtime_section, path, None

            return data, path, None

    return {}, None, None


def _parse_runtime_config(config: Dict[str, Any], config_path: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    if not config:
        return {}, None

    parsed: Dict[str, Any] = {}

    if "command" in config:
        command_value = config["command"]
        if isinstance(command_value, str):
            command_value = command_value.strip()
            if not command_value:
                return {}, f"Runtime config {config_path} field 'command' cannot be empty"
            parsed["command"] = command_value
        elif isinstance(command_value, (list, tuple)):
            command_list = []
            for item in command_value:
                if isinstance(item, (str, int, float)):
                    command_list.append(str(item))
                else:
                    return {}, f"Runtime config {config_path} field 'command' list items must be strings"
            if not command_list:
                return {}, f"Runtime config {config_path} field 'command' list cannot be empty"
            parsed["command"] = command_list
        else:
            return {}, f"Runtime config {config_path} field 'command' must be a string or list"

    if "timeout" in config:
        try:
            parsed["timeout"] = max(1, int(config["timeout"]))
        except (TypeError, ValueError):
            return {}, f"Runtime config {config_path} field 'timeout' must be an integer"

    for boolean_key in ["skip_install", "force_reinstall", "shell"]:
        if boolean_key in config:
            parsed_bool = _parse_bool(config[boolean_key])
            if parsed_bool is None:
                return {}, f"Runtime config {config_path} field '{boolean_key}' must be a boolean"
            parsed[boolean_key] = parsed_bool

    if "max_log_chars" in config:
        try:
            parsed["max_log_chars"] = max(0, int(config["max_log_chars"]))
        except (TypeError, ValueError):
            return {}, f"Runtime config {config_path} field 'max_log_chars' must be an integer"

    if "env" in config:
        env_values = config["env"]
        if not isinstance(env_values, dict):
            return {}, f"Runtime config {config_path} field 'env' must be an object"
        normalized_env: Dict[str, str] = {}
        for key, value in env_values.items():
            if not isinstance(key, str):
                return {}, f"Runtime config {config_path} field 'env' keys must be strings"
            normalized_env[key] = "" if value is None else str(value)
        parsed["env"] = normalized_env

    for key in ["cwd", "working_dir"]:
        if key in config:
            if not isinstance(config[key], str):
                return {}, f"Runtime config {config_path} field '{key}' must be a string"
            parsed[key] = config[key]

    return parsed, None


def _resolve_working_directory(
    project_root: str,
    cwd_value: Optional[str],
    config_path: Optional[str],
) -> Tuple[str, Optional[str]]:
    if not cwd_value:
        return project_root, None

    candidate = cwd_value
    if not os.path.isabs(candidate):
        candidate = os.path.join(project_root, cwd_value)

    if os.path.isdir(candidate):
        return candidate, None

    location = f" in {config_path}" if config_path else ""
    return project_root, f"Runtime config{location} references missing working directory '{cwd_value}'"


def _resolve_runtime_settings(project_root: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    print("ℹ️ Resolving runtime settings...")
    config_raw, config_path, config_error = _load_runtime_config(project_root)
    if config_error:
        return None, config_error

    parsed_config, parse_error = _parse_runtime_config(config_raw, config_path)
    if parse_error:
        return None, parse_error

    env_command = os.getenv("ML_UPGRADER_RUNTIME_COMMAND")
    command: Union[str, List[str], None] = env_command or parsed_config.get("command")
    if not command:
        return None, None

    command_label = "environment variable ML_UPGRADER_RUNTIME_COMMAND" if env_command else None
    if command_label is None and config_path:
        command_label = f"runtime config {config_path}"

    runtime_cwd_value = parsed_config.get("cwd") or parsed_config.get("working_dir")
    runtime_cwd, cwd_error = _resolve_working_directory(project_root, runtime_cwd_value, config_path)
    if cwd_error:
        return None, cwd_error

    settings = {
        "command": command,
        "command_label": command_label,
        "timeout": _runtime_timeout(parsed_config.get("timeout")),
        "skip_install": _resolve_bool_option(
            "ML_UPGRADER_RUNTIME_SKIP_INSTALL",
            parsed_config.get("skip_install"),
            False,
        ),
        "force_reinstall": _resolve_bool_option(
            "ML_UPGRADER_FORCE_REINSTALL",
            parsed_config.get("force_reinstall"),
            False,
        ),
        "log_limit": _max_runtime_log_chars(parsed_config.get("max_log_chars")),
        "env": parsed_config.get("env", {}),
        "cwd": runtime_cwd,
        "shell_preference": parsed_config.get("shell"),
    }

    return settings, None


def _prepare_command(
    command: Union[str, List[str]],
    shell_preference: Optional[bool],
) -> Tuple[Union[str, List[str]], bool]:
    if isinstance(command, str):
        use_shell = True if shell_preference is None else bool(shell_preference)
        return command, use_shell
    if isinstance(command, (list, tuple)):
        normalized = [str(part) for part in command]
        use_shell = False if shell_preference is None else bool(shell_preference)
        return normalized, use_shell
    raise ValueError("Runtime command must be a string or a list of strings")


def _discover_script_path(script: str, runtime_directory: str, project_root: str) -> Optional[str]:
    if not script or os.path.isabs(script):
        return None

    runtime_candidate = os.path.join(runtime_directory, script)
    if os.path.isfile(runtime_candidate):
        return script

    project_candidate = os.path.join(project_root, script)
    if os.path.isfile(project_candidate):
        return os.path.relpath(project_candidate, runtime_directory)

    filename = os.path.basename(script)
    matches: List[str] = []
    for root, dirnames, files in os.walk(project_root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in _RUNTIME_SEARCH_SKIP_DIRS and not name.startswith(".")
        ]
        if filename in files:
            matches.append(os.path.join(root, filename))
            if len(matches) > 1:
                return None

    if len(matches) == 1:
        return os.path.relpath(matches[0], runtime_directory)

    return None


def _auto_adjust_python_script_command(
    command: Union[str, List[str]],
    runtime_directory: str,
    project_root: str,
) -> Tuple[Union[str, List[str]], Optional[str]]:
    if not isinstance(command, list) or not command:
        return command, None

    executable = os.path.basename(command[0])
    python_names = {
        "python",
        "python3",
        os.path.basename(sys.executable),
    }
    if executable not in python_names:
        return command, None

    script_index: Optional[int] = None
    for idx in range(1, len(command)):
        part = command[idx]
        if part == "-m":
            # Module execution; no script path to adjust
            return command, None
        if part.startswith("-"):
            continue
        if part.endswith(".py"):
            script_index = idx
        break

    if script_index is None:
        return command, None

    script_arg = command[script_index]
    candidate = _discover_script_path(script_arg, runtime_directory, project_root)
    if candidate is None or candidate == script_arg:
        return command, None

    adjusted = command.copy()
    adjusted[script_index] = candidate
    note = (
        f"Runtime command script '{script_arg}' not found in '{runtime_directory}'. "
        f"Using '{candidate}' instead."
    )
    return adjusted, note


def _extract_missing_distribution(message: str) -> Optional[str]:
    if not message:
        return None
    match = re.search(r"No matching distribution found for ([A-Za-z0-9_.-]+)", message)
    if not match:
        return None
    spec = match.group(1)
    parts = re.split(r"[<>=!~]", spec, maxsplit=1)
    package = parts[0].strip()
    return package or None


def _extract_missing_module(log: str) -> Optional[str]:
    if not log:
        return None
    match = re.search(r"No module named ['\"]([A-Za-z0-9_.-]+)['\"]", log)
    if not match:
        return None
    module = match.group(1)
    if not module or module.startswith("_"):
        return None
    return module.split(".")[0]


def _discover_requirement_files(
    project_root: str,
    runtime_directory: str,
    command: Union[str, List[str]],
) -> List[str]:
    candidates: List[str] = []

    def _add(path: str) -> None:
        absolute = os.path.abspath(path)
        if os.path.isfile(absolute) and absolute not in candidates:
            candidates.append(absolute)

    _add(os.path.join(project_root, "requirements.txt"))

    if runtime_directory and runtime_directory != project_root:
        _add(os.path.join(runtime_directory, "requirements.txt"))

    if isinstance(command, list) and len(command) >= 2:
        script_path: Optional[str] = None
        for part in command[1:]:
            if part == "-m":
                script_path = None
                break
            if part.startswith("-"):
                continue
            if part.endswith(".py"):
                script_path = part
                break
        if script_path:
            candidate_path = script_path
            if not os.path.isabs(candidate_path):
                candidate_path = os.path.join(runtime_directory, candidate_path)
            script_dir = os.path.dirname(os.path.abspath(candidate_path))
            if script_dir:
                _add(os.path.join(script_dir, "requirements.txt"))

    return candidates


def _run_runtime_validation(
    project_root: str,
    command: Union[str, List[str]],
    *,
    timeout: int,
    skip_install: bool,
    force_reinstall: bool,
    log_limit: int,
    extra_env: Optional[Dict[str, str]],
    runtime_cwd: Optional[str],
    shell_preference: Optional[bool],
    command_label: Optional[str],
) -> Tuple[bool, Optional[str]]:
    if not project_root or not os.path.isdir(project_root):
        label_suffix = f" ({command_label})" if command_label else ""
        return False, f"Runtime validation error{label_suffix}: project root '{project_root}' not found"

    prepared_command, use_shell = _prepare_command(command, shell_preference)
    runtime_directory = runtime_cwd or project_root
    command_repr = _stringify_command(prepared_command, use_shell)

    if not os.path.isdir(runtime_directory):
        logs: List[Dict[str, Any]] = []
        logs.append({
            "step": "runtime_command",
            "command": "skip",
            "returncode": 1,
            "stdout": "",
            "stderr": f"Runtime directory '{runtime_directory}' not found",
            "timed_out": False,
        })
        reason = f"Runtime directory '{runtime_directory}' not found"
        return False, _format_runtime_error(command_repr, logs, reason, log_limit)

    venv_path = _select_venv_path(project_root)
    venv_bin, venv_python, _ = _resolve_venv_paths(venv_path)

    logs: List[Dict[str, Any]] = []

    if not os.path.isfile(venv_python):
        create_result = _run_subprocess([sys.executable, "-m", "venv", venv_path], cwd=project_root)
        logs.append(_step_log("create_virtualenv", create_result))
        if create_result["timed_out"]:
            reason = "Virtual environment creation timed out"
            return False, _format_runtime_error(command_repr, logs, reason, log_limit)
        if create_result["returncode"] != 0:
            reason = "Failed to create virtual environment"
            return False, _format_runtime_error(command_repr, logs, reason, log_limit)

    adjusted_command, adjustment_note = _auto_adjust_python_script_command(
        prepared_command,
        runtime_directory=runtime_directory,
        project_root=project_root,
    )
    if adjustment_note:
        print(f"ℹ️ {adjustment_note}")
    prepared_command = adjusted_command
    command_repr = _stringify_command(prepared_command, use_shell)

    requirement_files = _discover_requirement_files(project_root, runtime_directory, prepared_command)

    install_ok = True
    if not skip_install:
        install_ok = _ensure_dependencies_installed(
            project_root,
            venv_path,
            venv_python,
            logs,
            requirement_files=requirement_files,
            force_reinstall=force_reinstall,
        )
    else:
        logs.append({
            "step": "dependency_install",
            "command": "skip",
            "returncode": 0,
            "stdout": "Dependency installation skipped by configuration",
            "stderr": "",
            "timed_out": False,
        })

    base_env = _build_base_env(project_root, extra_env)
    env = base_env.copy()
    env["VIRTUAL_ENV"] = venv_path
    env["PATH"] = _prepend_to_path(venv_bin, base_env.get("PATH", ""))

    runtime_result = _run_subprocess(
        prepared_command,
        cwd=runtime_directory,
        env=env,
        timeout=timeout,
        shell=use_shell,
    )
    logs.append(_step_log("runtime_command", runtime_result))

    if install_ok and runtime_result["returncode"] == 0 and not runtime_result["timed_out"]:
        return True, None

    if (
        install_ok
        and not skip_install
        and not runtime_result["timed_out"]
        and runtime_result["returncode"]
    ):
        missing_module = _extract_missing_module(runtime_result.get("stderr", "") or runtime_result.get("stdout", ""))
        if missing_module:
            auto_install_cmd = [venv_python, "-m", "pip", "install", missing_module]
            auto_install_result = _run_subprocess(auto_install_cmd, cwd=project_root)
            logs.append(_step_log(f"auto_install:{missing_module}", auto_install_result))
            if auto_install_result["returncode"] == 0 and not auto_install_result["timed_out"]:
                retry_result = _run_subprocess(
                    prepared_command,
                    cwd=runtime_directory,
                    env=env,
                    timeout=timeout,
                    shell=use_shell,
                )
                logs.append(_step_log("runtime_command_retry", retry_result))
                if retry_result["returncode"] == 0 and not retry_result["timed_out"]:
                    return True, None
                runtime_result = retry_result

    label_suffix = f" ({command_label})" if command_label else ""
    if runtime_result["timed_out"]:
        reason = f"Runtime command timed out after {timeout} seconds{label_suffix}"
    elif runtime_result["returncode"]:
        reason = f"Runtime command exited with status {runtime_result['returncode']}{label_suffix}"
    elif not install_ok:
        reason = f"Dependency installation failed{label_suffix}"
    else:
        reason = f"Runtime validation failed{label_suffix}"

    return False, _format_runtime_error(command_repr, logs, reason, log_limit)


def _ensure_dependencies_installed(
    project_root: str,
    venv_path: str,
    venv_python: str,
    logs: List[Dict[str, Any]],
    *,
    requirement_files: Optional[List[str]] = None,
    force_reinstall: bool,
) -> bool:
    marker_path = os.path.join(venv_path, "ml_upgrader_marker.json")
    current_marker = _load_marker(marker_path) or {}

    requirement_files = requirement_files or []
    default_requirements = os.path.join(project_root, "requirements.txt")
    if os.path.isfile(default_requirements):
        requirement_files.insert(0, default_requirements)

    seen: set[str] = set()
    normalized_files: List[str] = []
    for path in requirement_files:
        absolute = os.path.abspath(path)
        if absolute in seen:
            continue
        seen.add(absolute)
        if os.path.isfile(absolute):
            normalized_files.append(absolute)

    requirement_hashes: Dict[str, Optional[str]] = {
        path: _hash_file(path) for path in normalized_files
    }
    marker_hashes = current_marker.get("requirements_hashes", {})

    install_needed = force_reinstall or requirement_hashes != marker_hashes

    setup_present = os.path.isfile(os.path.join(project_root, "setup.py"))
    pyproject_present = os.path.isfile(os.path.join(project_root, "pyproject.toml"))
    editable_sources_present = setup_present or pyproject_present
    editable_installed = bool(current_marker.get("editable_installed"))
    run_editable = editable_sources_present and (force_reinstall or not editable_installed)

    if not install_needed and not run_editable:
        logs.append({
            "step": "dependency_install",
            "command": "skip",
            "returncode": 0,
            "stdout": "Dependencies already up to date",
            "stderr": "",
            "timed_out": False,
        })
        return True

    toolchain_cmd = [venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    toolchain_result = _run_subprocess(toolchain_cmd, cwd=project_root)
    logs.append(_step_log("upgrade_toolchain", toolchain_result))
    if toolchain_result["timed_out"] or toolchain_result["returncode"] != 0:
        return False

    success = True
    if install_needed and normalized_files:
        for req_path in normalized_files:
            install_result = _run_subprocess(
                [venv_python, "-m", "pip", "install", "-r", req_path],
                cwd=os.path.dirname(req_path) or project_root,
            )
            step_name = f"dependency_install:{os.path.relpath(req_path, project_root)}"
            logs.append(_step_log(step_name, install_result))
            if install_result["timed_out"]:
                success = False
                break
            if install_result["returncode"] != 0:
                fallback_package = _extract_missing_distribution(install_result.get("stderr", ""))
                if fallback_package:
                    fallback_cmd = [venv_python, "-m", "pip", "install", fallback_package]
                    fallback_result = _run_subprocess(fallback_cmd, cwd=project_root)
                    logs.append(_step_log(f"dependency_fallback:{fallback_package}", fallback_result))
                    if fallback_result["timed_out"] or fallback_result["returncode"] != 0:
                        success = False
                        break
                    continue
                success = False
                break
    elif install_needed and not normalized_files:
        logs.append({
            "step": "dependency_install",
            "command": "skip",
            "returncode": 0,
            "stdout": "No requirements files discovered",
            "stderr": "",
            "timed_out": False,
        })

    if run_editable and success:
        editable_result = _run_subprocess([venv_python, "-m", "pip", "install", "-e", project_root], cwd=project_root)
        logs.append(_step_log("editable_install", editable_result))
        if editable_result["timed_out"] or editable_result["returncode"] != 0:
            success = False

    if success:
        marker_data = {
            "requirements_hashes": requirement_hashes,
            "editable_installed": run_editable or editable_installed,
        }
        _save_marker(marker_path, marker_data)

    return success


def _build_base_env(project_root: str, extra_env: Optional[Dict[str, str]]) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_to_path(project_root, env.get("PYTHONPATH", ""))
    if extra_env:
        for key, value in extra_env.items():
            env[str(key)] = value
    return env


def _read_requirements_lines(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    lines: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line.strip():
                    lines.append(line)
    except OSError:
        return []
    return lines


def _extract_requirement_packages(path: str) -> List[str]:
    packages: List[str] = []
    for line in _read_requirements_lines(path):
        name = _normalize_requirement_name(line)
        if name:
            packages.append(name)
    return packages


def _normalize_requirement_name(line: str) -> Optional[str]:
    stripped = line.split("#", 1)[0].strip()
    if not stripped:
        return None
    match = re.match(r"[A-Za-z0-9._-]+", stripped)
    if not match:
        return None
    return match.group(0)


def _resolve_venv_paths(venv_path: str) -> Tuple[str, str, str]:
    if os.name == "nt":
        bin_dir = os.path.join(venv_path, "Scripts")
        python_path = os.path.join(bin_dir, "python.exe")
        pip_path = os.path.join(bin_dir, "pip.exe")
    else:
        bin_dir = os.path.join(venv_path, "bin")
        python_path = os.path.join(bin_dir, "python")
        pip_path = os.path.join(bin_dir, "pip")
    return bin_dir, python_path, pip_path


def _run_subprocess(
    cmd: Any,
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    shell: bool = False,
) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=shell,
        )
        return {
            "command": _stringify_command(cmd, shell),
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": _stringify_command(cmd, shell),
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
            "timeout": timeout,
        }


def _stringify_command(cmd: Any, shell: bool) -> str:
    if shell:
        return str(cmd)
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _step_log(step: str, result: Dict[str, Any]) -> Dict[str, Any]:
    entry = result.copy()
    entry["step"] = step
    return entry


def _format_runtime_error(command: str, logs: List[Dict[str, Any]], reason: str, log_limit: int) -> str:
    parts = [reason, f"Runtime command: {command}"]
    for entry in logs:
        header = "--- Step: {}".format(entry.get("step", "unknown"))
        if entry.get("timed_out"):
            timeout_value = entry.get("timeout")
            if timeout_value:
                header += f" (timed out after {timeout_value} seconds)"
            else:
                header += " (timed out)"
        else:
            header += f" (exit {entry.get('returncode')})"
        parts.append(header)
        if entry.get("command"):
            parts.append(f"Command: {entry['command']}")
        stdout = (entry.get("stdout") or "").strip()
        if stdout:
            parts.append("stdout:\n" + _truncate_log(stdout, log_limit))
        stderr = (entry.get("stderr") or "").strip()
        if stderr:
            parts.append("stderr:\n" + _truncate_log(stderr, log_limit))
    return "\n".join(part for part in parts if part)


def _truncate_log(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    remainder = len(content) - limit
    return f"{content[:limit]}\n... (truncated {remainder} characters)"


def _prepend_to_path(value: str, existing: str) -> str:
    if not value:
        return existing
    if not existing:
        return value
    return f"{value}{os.pathsep}{existing}"


def _hash_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_marker(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _save_marker(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
    except OSError:
        pass
