import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

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

def upgrade_repo(old_repo: str, new_repo: str, dependency_overrides: Optional[Dict[str, str]] = None) -> str:
    """
    Upgrade entire repository with comprehensive reporting
    Returns path to generated report
    """

    previous_project_root = os.getenv("ML_UPGRADER_PROJECT_ROOT")
    os.environ["ML_UPGRADER_PROJECT_ROOT"] = new_repo

    try:
        # Initialize components
        report_generator_instance = report_generator.UpgradeReportGenerator()
        dependency_updater_instance = dependency_upgrader.DependencyUpdater(dependency_overrides)
        
        # Setup output directory
        if os.path.exists(new_repo):
            shutil.rmtree(new_repo)

        def _ignore_env_dirs(_, names):
            ignored = []
            if ".venv" in names:
                ignored.append(".venv")
            return ignored

        shutil.copytree(old_repo, new_repo, ignore=_ignore_env_dirs)
        
        print(f"Starting repo upgrade: {old_repo} → {new_repo}")
        
        # Update dependencies
        print("📦 Updating dependencies...")
        dependency_updater_instance.update_requirements_txt(new_repo)
        dependency_updater_instance.update_setup_py(new_repo)
        report_generator_instance.add_dependency_changes(dependency_updater_instance.updated_deps)
        
        # Upgrade Python files
        python_files: List[str] = []
        for root, _, files in os.walk(new_repo):
            for f in files:
                if f.endswith(".py"):
                    python_files.append(os.path.join(root, f))
        
        print(f"🔄 Upgrading {len(python_files)} Python files...")
        
        for file_path in python_files:
            try:
                # Skip __pycache__ and other generated files
                if '__pycache__' in file_path or '.pyc' in file_path:
                    continue

                filename = os.path.basename(file_path)
                rel_path = os.path.relpath(file_path, new_repo)
                if filename.startswith('._'):
                    print(f"ℹ️ Skipping macOS resource fork file: {rel_path}")
                    continue
                if os.sep + '__MACOSX' + os.sep in file_path or rel_path.startswith('__MACOSX'):
                    print(f"ℹ️ Skipping macOS metadata file: {rel_path}")
                    continue

                result = agentic_upgrader.upgrade_file(file_path, file_path)
                report_generator_instance.add_file_result(result)
                
            except Exception as e:
                print(f"⚠️ Error upgrading {file_path}: {e}")
                # Add failed result to report
                result = report_generator.FileUpgradeResult(
                    file_path=file_path,
                    success=False,
                    attempts=0,
                    api_changes=[],
                    error=str(e)
                )
                report_generator_instance.add_file_result(result)
        
        # Generate report
        report_path = os.path.join(new_repo, "UPGRADE_REPORT.md")
        report_generator_instance.generate_report(report_path)
        
        successful = len([r for r in report_generator_instance.results if r.success])
        total = len(report_generator_instance.results)

        if total > 0 and successful == total:
            print("🧪 Running runtime validation across upgraded project...")
            runtime_ok, runtime_error = validator.validate_repository(new_repo)
            if runtime_ok:
                print("✅ Runtime validation passed")
            else:
                raise RuntimeError(f"Runtime validation failed: {runtime_error}")
        else:
            print("⚠️ Skipping runtime validation because some files failed validation")
        
        print(f"✅ Upgrade complete! {successful}/{total} files upgraded successfully")
        print(f"📄 Report generated: {report_path}")
        
        return report_path
    finally:
        if previous_project_root is None:
            os.environ.pop("ML_UPGRADER_PROJECT_ROOT", None)
        else:
            os.environ["ML_UPGRADER_PROJECT_ROOT"] = previous_project_root
