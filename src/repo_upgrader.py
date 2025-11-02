import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

if __package__ is None or __package__ == "":
    _CURRENT_DIR = Path(__file__).resolve().parent
    _ROOT_DIR = _CURRENT_DIR.parent
    _ROOT_STR = str(_ROOT_DIR)
    if _ROOT_STR not in sys.path:
        sys.path.insert(0, _ROOT_STR)
    import src.agentic_upgrader as agentic_upgrader  # type: ignore
    import src.dependency_upgrader as dependency_upgrader  # type: ignore
    import src.report_generator as report_generator  # type: ignore
    import src.validator as validator  # type: ignore
else:
    from . import agentic_upgrader, dependency_upgrader, report_generator, validator


def _ignore_env_dirs(_: str, names: List[str]) -> List[str]:
    ignored = []
    if ".venv" in names:
        ignored.append(".venv")
    if ".ml-upgrader" in names:
        ignored.append(".ml-upgrader")
    return ignored

ProgressCallback = Callable[[str, Optional[float]], None]


def upgrade_repo(
    old_repo: str,
    new_repo: str,
    dependency_overrides: Optional[Dict[str, str]] = None,
    *,
    verify_runtime_outputs: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """
    Upgrade entire repository with comprehensive reporting
    Returns path to generated report
    """

    previous_project_root = os.getenv("ML_UPGRADER_PROJECT_ROOT")
    baseline_output_dir: Optional[str] = None
    upgraded_output_dir: Optional[str] = None
    baseline_outputs_captured = False
    baseline_temp_dir: Optional[str] = None

    def _emit_status(message: str, progress: Optional[float] = None, *, console: bool = True) -> None:
        if console:
            print(message)
        if progress_callback is not None:
            try:
                progress_callback(message, progress)
            except Exception as exc:
                print(f"⚠️ Progress callback error: {exc}")

    try:
        _emit_status("🚀 Preparing repository upgrade...", progress=0, console=False)

        if verify_runtime_outputs:
            baseline_output_dir = os.path.join(old_repo, ".ml-upgrader", "runtime_outputs", "baseline")
            upgraded_output_dir = os.path.join(new_repo, ".ml-upgrader", "runtime_outputs", "upgraded")
            baseline_temp_dir = tempfile.mkdtemp(prefix="ml-upgrader-baseline-")
            baseline_project = os.path.join(baseline_temp_dir, "baseline_repo")

            if os.path.isdir(baseline_output_dir):
                shutil.rmtree(baseline_output_dir)

            _emit_status("🧪 Running baseline runtime validation...", progress=5)
            shutil.copytree(old_repo, baseline_project, ignore=_ignore_env_dirs)
            os.environ["ML_UPGRADER_PROJECT_ROOT"] = baseline_project

            baseline_ok, baseline_error = validator.perform_project_runtime_validation(
                baseline_project,
                output_capture_dir=baseline_output_dir,
            )
            if not baseline_ok:
                message = baseline_error or "unknown error"
                raise RuntimeError(f"Baseline runtime validation failed: {message}")

            baseline_outputs_captured = os.path.isdir(baseline_output_dir)
            if baseline_outputs_captured:
                _emit_status(f"✅ Baseline runtime outputs saved to {baseline_output_dir}", progress=15)
            else:
                _emit_status("ℹ️ Baseline runtime validation skipped (no runtime command configured).", progress=15)
        else:
            _emit_status(
                "ℹ️ Skipping baseline runtime validation (runtime output comparison disabled).",
                progress=10,
            )

        os.environ["ML_UPGRADER_PROJECT_ROOT"] = new_repo

        # Initialize components
        report_generator_instance = report_generator.UpgradeReportGenerator()
        dependency_updater_instance = dependency_upgrader.DependencyUpdater(dependency_overrides)

        # Setup output directory
        if os.path.exists(new_repo):
            shutil.rmtree(new_repo)

        _emit_status("📁 Preparing copy of repository...", progress=20, console=False)
        shutil.copytree(old_repo, new_repo, ignore=_ignore_env_dirs)

        if verify_runtime_outputs and upgraded_output_dir and os.path.isdir(upgraded_output_dir):
            shutil.rmtree(upgraded_output_dir)

        _emit_status(f"Starting repo upgrade: {old_repo} → {new_repo}", progress=25)

        # Update dependencies
        _emit_status("📦 Updating dependencies...", progress=30)
        dependency_updater_instance.update_requirements_txt(new_repo)
        dependency_updater_instance.update_setup_py(new_repo)
        report_generator_instance.add_dependency_changes(dependency_updater_instance.updated_deps)
        _emit_status("✅ Dependencies updated.", progress=35, console=False)

        # Upgrade Python files
        python_files: List[str] = []
        for root, _, files in os.walk(new_repo):
            for f in files:
                if f.endswith(".py"):
                    python_files.append(os.path.join(root, f))

        total_python_files = len(python_files)
        _emit_status(f"🔄 Upgrading {total_python_files} Python files...", progress=40)

        for index, file_path in enumerate(python_files, start=1):
            try:
                # Skip __pycache__ and other generated files
                if '__pycache__' in file_path or '.pyc' in file_path:
                    continue

                filename = os.path.basename(file_path)
                rel_path = os.path.relpath(file_path, new_repo)
                if filename.startswith('._'):
                    _emit_status(f"ℹ️ Skipping macOS resource fork file: {rel_path}", console=True)
                    continue
                if os.sep + '__MACOSX' + os.sep in file_path or rel_path.startswith('__MACOSX'):
                    _emit_status(f"ℹ️ Skipping macOS metadata file: {rel_path}", console=True)
                    continue

                if total_python_files:
                    progress = 40 + (index / total_python_files) * 40
                else:
                    progress = 80
                _emit_status(f"🛠️ Upgrading {rel_path} ({index}/{total_python_files})...", progress=progress, console=False)

                result = agentic_upgrader.upgrade_file(file_path, file_path)
                report_generator_instance.add_file_result(result)

            except Exception as e:
                _emit_status(f"⚠️ Error upgrading {file_path}: {e}")
                # Add failed result to report
                result = report_generator.FileUpgradeResult(
                    file_path=file_path,
                    success=False,
                    attempts=0,
                    api_changes=[],
                    error=str(e)
                )
                report_generator_instance.add_file_result(result)
        if not total_python_files:
            _emit_status("ℹ️ No Python files found to upgrade.", progress=80)
        else:
            _emit_status("✅ Python files processed.", progress=82, console=False)

        # Generate report
        report_path = os.path.join(new_repo, "UPGRADE_REPORT.md")
        _emit_status("📄 Generating upgrade report...", progress=85, console=False)
        report_generator_instance.generate_report(report_path)

        successful = len([r for r in report_generator_instance.results if r.success])
        total = len(report_generator_instance.results)

        if total > 0 and successful == total:
            _emit_status("🧪 Running runtime validation across upgraded project...", progress=90)
            if verify_runtime_outputs and upgraded_output_dir and os.path.isdir(upgraded_output_dir):
                shutil.rmtree(upgraded_output_dir)
            runtime_ok, runtime_error = validator.validate_repository(
                new_repo,
                runtime_output_dir=upgraded_output_dir if verify_runtime_outputs else None,
                runtime_compare_dir=baseline_output_dir if verify_runtime_outputs and baseline_outputs_captured else None,
            )
            if runtime_ok:
                if verify_runtime_outputs and upgraded_output_dir and os.path.isdir(upgraded_output_dir):
                    _emit_status(
                        f"✅ Runtime validation passed; outputs saved to {upgraded_output_dir}",
                        progress=95,
                    )
                else:
                    _emit_status("✅ Runtime validation passed", progress=95)
            else:
                raise RuntimeError(f"Runtime validation failed: {runtime_error}")
        else:
            _emit_status("⚠️ Skipping runtime validation because some files failed validation", progress=95)

        _emit_status(f"✅ Upgrade complete! {successful}/{total} files upgraded successfully", progress=100)
        _emit_status(f"📄 Report generated: {report_path}", progress=100)

        return report_path
    finally:
        if baseline_temp_dir and os.path.isdir(baseline_temp_dir):
            shutil.rmtree(baseline_temp_dir, ignore_errors=True)
        if previous_project_root is None:
            os.environ.pop("ML_UPGRADER_PROJECT_ROOT", None)
        else:
            os.environ["ML_UPGRADER_PROJECT_ROOT"] = previous_project_root
