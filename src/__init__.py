"""
ML Repository Upgrader

An intelligent tool for automatically upgrading legacy ML repositories
to use the latest APIs for TensorFlow, PyTorch, NumPy, JAX, and other ML libraries.
"""

__version__ = "1.0.0"

from .repo_upgrader import upgrade_repo
from .agentic_upgrader import upgrade_file
from .report_generator import UpgradeReportGenerator, FileUpgradeResult

__all__ = [
    "upgrade_repo",
    "upgrade_file", 
    "UpgradeReportGenerator",
    "FileUpgradeResult"
]
