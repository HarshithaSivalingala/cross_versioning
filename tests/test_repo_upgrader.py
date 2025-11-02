import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.repo_upgrader as repo_upgrader


class DummyDependencyUpdater:
    def __init__(self, _overrides):
        self.updated_deps = []

    def update_requirements_txt(self, _repo_path: str) -> None:
        return None

    def update_setup_py(self, _repo_path: str) -> None:
        return None


class DummyReportGenerator:
    def __init__(self):
        self.results = []
        self.dependency_changes = []

    def add_dependency_changes(self, changes):
        if changes:
            self.dependency_changes.extend(changes)

    def add_file_result(self, _result):
        self.results.append(SimpleNamespace(success=True))

    def generate_report(self, report_path: str) -> None:
        Path(report_path).write_text("# report\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def _monkeypatch_dependencies(monkeypatch):
    monkeypatch.setattr(repo_upgrader.dependency_upgrader, "DependencyUpdater", DummyDependencyUpdater)
    monkeypatch.setattr(repo_upgrader.report_generator, "UpgradeReportGenerator", DummyReportGenerator)
    monkeypatch.setattr(
        repo_upgrader.agentic_upgrader,
        "upgrade_file",
        lambda input_path, output_path: SimpleNamespace(
            file_path=output_path,
            success=True,
            attempts=1,
            api_changes=[],
            error=None,
        ),
    )


def _fake_runtime_functions(monkeypatch):
    baseline_calls = []
    validate_calls = []

    def fake_perform(project_root: str, *, output_capture_dir=None, compare_with_dir=None):
        baseline_calls.append(
            {
                "project_root": project_root,
                "output_capture_dir": output_capture_dir,
                "compare_with_dir": compare_with_dir,
            }
        )
        if output_capture_dir:
            os.makedirs(output_capture_dir, exist_ok=True)
            Path(output_capture_dir, "stdout.txt").write_text("baseline\n", encoding="utf-8")
            Path(output_capture_dir, "stderr.txt").write_text("", encoding="utf-8")
            Path(output_capture_dir, "metadata.json").write_text(
                json.dumps(
                    {
                        "project_root": project_root,
                        "success": True,
                        "returncode": 0,
                        "timed_out": False,
                    }
                ),
                encoding="utf-8",
            )
            Path(output_capture_dir, "logs.json").write_text("[]", encoding="utf-8")
        return True, None

    def fake_validate(project_root: str, *, runtime_output_dir=None, runtime_compare_dir=None):
        validate_calls.append(
            {
                "project_root": project_root,
                "runtime_output_dir": runtime_output_dir,
                "runtime_compare_dir": runtime_compare_dir,
            }
        )
        if runtime_output_dir:
            os.makedirs(runtime_output_dir, exist_ok=True)
            Path(runtime_output_dir, "stdout.txt").write_text("baseline\n", encoding="utf-8")
            Path(runtime_output_dir, "stderr.txt").write_text("", encoding="utf-8")
            Path(runtime_output_dir, "metadata.json").write_text(
                json.dumps(
                    {
                        "project_root": project_root,
                        "success": True,
                        "returncode": 0,
                        "timed_out": False,
                    }
                ),
                encoding="utf-8",
            )
            Path(runtime_output_dir, "logs.json").write_text("[]", encoding="utf-8")
        return True, None

    monkeypatch.setattr(repo_upgrader.validator, "perform_project_runtime_validation", fake_perform)
    monkeypatch.setattr(repo_upgrader.validator, "validate_repository", fake_validate)

    return baseline_calls, validate_calls


def _create_minimal_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "module.py").write_text("print('hello world')\n", encoding="utf-8")


def test_upgrade_repo_runs_baseline_when_verification_enabled(tmp_path, monkeypatch):
    old_repo = tmp_path / "old"
    new_repo = tmp_path / "new"
    _create_minimal_repo(old_repo)

    baseline_calls, validate_calls = _fake_runtime_functions(monkeypatch)
    report_path = repo_upgrader.upgrade_repo(
        str(old_repo),
        str(new_repo),
        verify_runtime_outputs=True,
    )

    assert os.path.exists(report_path)
    assert baseline_calls, "Expected baseline runtime validation to run"
    assert validate_calls, "Expected upgraded runtime validation to run"

    baseline_dir = baseline_calls[0]["output_capture_dir"]
    assert baseline_dir
    assert validate_calls[0]["runtime_compare_dir"] == baseline_dir


def test_upgrade_repo_skips_baseline_when_verification_disabled(tmp_path, monkeypatch):
    old_repo = tmp_path / "old"
    new_repo = tmp_path / "new"
    _create_minimal_repo(old_repo)

    baseline_calls, validate_calls = _fake_runtime_functions(monkeypatch)
    report_path = repo_upgrader.upgrade_repo(
        str(old_repo),
        str(new_repo),
        verify_runtime_outputs=False,
    )

    assert os.path.exists(report_path)
    assert baseline_calls == []
    assert validate_calls, "Runtime validation for upgraded repo should still run"
    assert validate_calls[0]["runtime_compare_dir"] is None
    assert validate_calls[0]["runtime_output_dir"] is None
