"""
ML Repository Upgrader

An intelligent tool for automatically upgrading legacy ML repositories
to use the latest APIs for TensorFlow, PyTorch, NumPy, JAX, and other ML libraries.
"""

__version__ = "1.0.0"

from typing import Any, Dict, Optional

__all__ = [
    "upgrade_repo",
    "upgrade_file",
    "UpgradeReportGenerator",
    "FileUpgradeResult",
]


def upgrade_repo(
    old_repo: str,
    new_repo: str,
    dependency_overrides: Optional[Dict[str, str]] = None,
) -> str:
    from .repo_upgrader import upgrade_repo as _upgrade_repo

    return _upgrade_repo(old_repo, new_repo, dependency_overrides)


def upgrade_file(input_path: str, output_path: str) -> Any:
    from .agentic_upgrader import upgrade_file as _upgrade_file

    return _upgrade_file(input_path, output_path)


def __getattr__(name: str) -> Any:
    if name == "UpgradeReportGenerator":
        from .report_generator import UpgradeReportGenerator

        return UpgradeReportGenerator
    if name == "FileUpgradeResult":
        from .report_generator import FileUpgradeResult

        return FileUpgradeResult
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
