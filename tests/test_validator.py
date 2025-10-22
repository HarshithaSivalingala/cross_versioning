import json
import os
import tempfile

import pytest

from src.validator import validate_syntax, validate_code, validate_repository

class TestValidator:
    
    def test_validate_syntax_valid(self):
        """Test syntax validation with valid Python code"""
        code = "import os\nprint('Hello, World!')"
        is_valid, error = validate_syntax(code)
        
        assert is_valid == True
        assert error is None
    
    def test_validate_syntax_invalid(self):
        """Test syntax validation with invalid Python code"""
        code = "import os\nprint('Hello, World!"  # Missing closing quote
        is_valid, error = validate_syntax(code)
        
        assert is_valid == False
        assert error is not None
        assert "syntax error" in error.lower()
    
    def test_validate_code_valid(self):
        """Test full code validation with valid file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("import sys\nprint('Hello, World!')")
            f.flush()
            
            is_valid, error = validate_code(f.name)
            
            # Clean up
            os.unlink(f.name)
            
            assert is_valid == True
            assert error is None
    
    def test_validate_code_syntax_error(self):
        """Test full code validation with syntax error"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("import sys\nprint('Hello, World!")  # Missing quote
            f.flush()
            
            is_valid, error = validate_code(f.name)
            
            # Clean up
            os.unlink(f.name)
            
            assert is_valid == False
            assert error is not None

    def test_validate_repository(self, tmp_path):
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        module_path = package_dir / "mod.py"
        module_path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        is_valid, error = validate_repository(str(tmp_path))

        assert is_valid is True
        assert error is None

    def test_validate_repository_adjusts_runtime_script(self, tmp_path, monkeypatch):
        project_root = tmp_path / "project"
        project_root.mkdir()

        runtime_package = project_root / "app"
        runtime_package.mkdir()
        (runtime_package / "main.py").write_text(
            "def main():\n    return 0\n\nif __name__ == '__main__':\n    main()\n",
            encoding="utf-8",
        )
        (runtime_package / "requirements.txt").write_text("tensorflow==2.0.0\n", encoding="utf-8")

        module_path = project_root / "module.py"
        module_path.write_text("VALUE = 1\n", encoding="utf-8")

        runtime_config = {
            "runtime": {
                "command": ["python", "main.py"],
                "skip_install": False,
            }
        }
        (project_root / "ml_upgrader_runtime.json").write_text(
            json.dumps(runtime_config),
            encoding="utf-8",
        )

        fake_venv = tmp_path / "fake_venv"
        (fake_venv / "bin").mkdir(parents=True)
        (fake_venv / "bin" / "python").write_text("", encoding="utf-8")

        import src.runtime_validation as runtime_validation  # noqa: WPS433

        monkeypatch.setattr(runtime_validation, "_select_venv_path", lambda _: str(fake_venv))
        monkeypatch.setattr(
            runtime_validation,
            "_resolve_venv_paths",
            lambda path: (os.path.join(path, "bin"), os.path.join(path, "bin", "python"), os.path.join(path, "bin", "pip")),
        )

        recorded_commands = []

        def fake_run_subprocess(cmd, *, cwd=None, env=None, timeout=None, shell=False):
            recorded_commands.append((cmd, cwd))
            return {
                "command": "",
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            }

        monkeypatch.setattr(runtime_validation, "_run_subprocess", fake_run_subprocess)

        is_valid, error = validate_repository(str(project_root))

        assert is_valid is True
        assert error is None
        assert recorded_commands, "Runtime command was not executed"

        pip_installs = [
            cmd for cmd, _ in recorded_commands if isinstance(cmd, list) and "-r" in cmd
        ]
        assert pip_installs, "Expected dependency installation command"
        assert any(
            arg.endswith(os.path.join("app", "requirements.txt"))
            for cmd in pip_installs
            for arg in cmd
        ), "Expected requirements install for script directory"

        runtime_cmd, runtime_cwd = recorded_commands[-1]
        assert runtime_cwd == str(project_root)
        assert isinstance(runtime_cmd, list)
        assert "app/main.py" in runtime_cmd

    def test_runtime_validation_auto_installs_missing_module(self, tmp_path, monkeypatch):
        project_root = tmp_path / "project"
        project_root.mkdir()

        (project_root / "main.py").write_text(
            "import missingpkg\n\nprint('hello')\n",
            encoding="utf-8",
        )

        runtime_config = {
            "runtime": {
                "command": ["python", "main.py"],
                "skip_install": False,
            }
        }
        (project_root / "ml_upgrader_runtime.json").write_text(
            json.dumps(runtime_config),
            encoding="utf-8",
        )

        fake_venv = tmp_path / "fake_venv"
        (fake_venv / "bin").mkdir(parents=True)
        fake_python = fake_venv / "bin" / "python"
        fake_python.write_text("", encoding="utf-8")

        import src.runtime_validation as runtime_validation  # noqa: WPS433

        monkeypatch.setattr(runtime_validation, "_select_venv_path", lambda _: str(fake_venv))
        monkeypatch.setattr(
            runtime_validation,
            "_resolve_venv_paths",
            lambda path: (os.path.join(path, "bin"), os.path.join(path, "bin", "python"), os.path.join(path, "bin", "pip")),
        )
        monkeypatch.setattr(runtime_validation, "_ensure_dependencies_installed", lambda *args, **kwargs: True)

        recorded_commands = []
        module_installed = {"missingpkg": False}

        def fake_run_subprocess(cmd, *, cwd=None, env=None, timeout=None, shell=False):
            cmd_list = cmd if isinstance(cmd, list) else [cmd]
            recorded_commands.append((cmd_list, cwd))

            if cmd_list and cmd_list[0] == fake_python.as_posix() and cmd_list[1:3] == ["-m", "pip"]:
                if "missingpkg" in cmd_list:
                    module_installed["missingpkg"] = True
                return {
                    "command": "",
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                }

            if cmd_list and cmd_list[0].endswith("python") and "main.py" in cmd_list:
                if module_installed["missingpkg"]:
                    return {
                        "command": "",
                        "returncode": 0,
                        "stdout": "done",
                        "stderr": "",
                        "timed_out": False,
                    }
                return {
                    "command": "",
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "ModuleNotFoundError: No module named 'missingpkg'",
                    "timed_out": False,
                }

            return {
                "command": "",
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            }

        monkeypatch.setattr(runtime_validation, "_run_subprocess", fake_run_subprocess)

        is_valid, error = validate_repository(str(project_root))

        assert is_valid is True
        assert error is None

        runtime_calls = [cmd for cmd, _ in recorded_commands if "main.py" in cmd]
        assert len(runtime_calls) >= 2, "Expected runtime command retry after auto install"

        auto_installs = [cmd for cmd, _ in recorded_commands if "missingpkg" in cmd]
        assert auto_installs, "Expected auto installation of missing module"

    def test_runtime_validation_runs_setup_commands(self, tmp_path, monkeypatch):
        project_root = tmp_path / "project"
        project_root.mkdir()

        (project_root / "main.py").write_text(
            "def main():\n    return 0\n\nif __name__ == '__main__':\n    main()\n",
            encoding="utf-8",
        )

        runtime_config = {
            "runtime": {
                "command": ["python", "main.py"],
                "setup_commands": [
                    "echo setup-one",
                    ["python", "-c", "print('setup-two')"],
                ],
                "skip_install": True,
            }
        }
        (project_root / "ml_upgrader_runtime.json").write_text(
            json.dumps(runtime_config),
            encoding="utf-8",
        )

        fake_venv = tmp_path / "fake_venv"
        (fake_venv / "bin").mkdir(parents=True)
        fake_python = fake_venv / "bin" / "python"
        fake_python.write_text("", encoding="utf-8")

        import src.runtime_validation as runtime_validation  # noqa: WPS433

        monkeypatch.setattr(runtime_validation, "_select_venv_path", lambda _: str(fake_venv))
        monkeypatch.setattr(
            runtime_validation,
            "_resolve_venv_paths",
            lambda path: (
                os.path.join(path, "bin"),
                os.path.join(path, "bin", "python"),
                os.path.join(path, "bin", "pip"),
            ),
        )

        recorded_commands = []

        def fake_run_subprocess(cmd, *, cwd=None, env=None, timeout=None, shell=False):
            recorded_commands.append((cmd, shell, cwd))
            return {
                "command": runtime_validation._stringify_command(cmd, shell),
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            }

        monkeypatch.setattr(runtime_validation, "_run_subprocess", fake_run_subprocess)

        is_valid, error = validate_repository(str(project_root))

        assert is_valid is True
        assert error is None
        assert len(recorded_commands) == 3

        setup_one, shell_one, cwd_one = recorded_commands[0]
        assert setup_one == "echo setup-one"
        assert shell_one is True
        assert cwd_one == str(project_root)

        setup_two, shell_two, cwd_two = recorded_commands[1]
        assert isinstance(setup_two, list)
        assert setup_two[:3] == ["python", "-c", "print('setup-two')"]
        assert shell_two is False
        assert cwd_two == str(project_root)

        runtime_cmd, runtime_shell, runtime_cwd = recorded_commands[2]
        assert isinstance(runtime_cmd, list)
        assert "main.py" in runtime_cmd
        assert runtime_shell is False
        assert runtime_cwd == str(project_root)

    def test_dependency_install_fallbacks_to_latest(self, tmp_path, monkeypatch):
        project_root = tmp_path / "project"
        project_root.mkdir()

        (project_root / "input").mkdir()
        requirements_path = project_root / "input" / "requirements.txt"
        requirements_path.write_text("tensorflow==1.15.0\n", encoding="utf-8")

        (project_root / "input" / "main.py").write_text("print('ok')\n", encoding="utf-8")

        runtime_config = {
            "runtime": {
                "command": ["python", "input/main.py"],
                "skip_install": False,
            }
        }
        (project_root / "ml_upgrader_runtime.json").write_text(
            json.dumps(runtime_config),
            encoding="utf-8",
        )

        fake_venv = tmp_path / "fake_venv"
        (fake_venv / "bin").mkdir(parents=True)
        fake_python = fake_venv / "bin" / "python"
        fake_python.write_text("", encoding="utf-8")
        fake_pip = fake_venv / "bin" / "pip"
        fake_pip.write_text("", encoding="utf-8")

        import src.runtime_validation as runtime_validation  # noqa: WPS433

        monkeypatch.setattr(runtime_validation, "_select_venv_path", lambda _: str(fake_venv))
        monkeypatch.setattr(
            runtime_validation,
            "_resolve_venv_paths",
            lambda path: (os.path.join(path, "bin"), os.path.join(path, "bin", "python"), os.path.join(path, "bin", "pip")),
        )

        commands = []
        fallback_invoked = {"tensorflow": False}

        def fake_run_subprocess(cmd, *, cwd=None, env=None, timeout=None, shell=False):
            cmd_list = cmd if isinstance(cmd, list) else [cmd]
            commands.append((cmd_list, cwd))

            if cmd_list[:4] == [fake_python.as_posix(), "-m", "pip", "install"] and "--upgrade" in cmd_list:
                return {
                    "command": "",
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                }

            if "-r" in cmd_list:
                return {
                    "command": "",
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "ERROR: No matching distribution found for tensorflow==1.15.0",
                    "timed_out": False,
                }

            if cmd_list[:5] == [fake_python.as_posix(), "-m", "pip", "install", "tensorflow"]:
                fallback_invoked["tensorflow"] = True
                return {
                    "command": "",
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                }

            if cmd_list and "input/main.py" in cmd_list:
                return {
                    "command": "",
                    "returncode": 0,
                    "stdout": "ok",
                    "stderr": "",
                    "timed_out": False,
                }

            return {
                "command": "",
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            }

        monkeypatch.setattr(runtime_validation, "_run_subprocess", fake_run_subprocess)

        is_valid, error = validate_repository(str(project_root))

        assert is_valid is True
        assert error is None
        assert fallback_invoked["tensorflow"] is True
        fallback_steps = [
            cmd for cmd, _ in commands if cmd[:5] == [fake_python.as_posix(), "-m", "pip", "install", "tensorflow"]
        ]
        assert fallback_steps, "Expected fallback installation command"
