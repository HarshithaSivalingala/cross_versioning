import os
import shutil
import agentic_upgrader
import dependency_upgrader
import report_generator
from typing import List

def upgrade_repo(old_repo: str, new_repo: str) -> str:
    """
    Upgrade entire repository with comprehensive reporting
    Returns path to generated report
    """
    
    previous_project_root = os.getenv("ML_UPGRADER_PROJECT_ROOT")
    os.environ["ML_UPGRADER_PROJECT_ROOT"] = new_repo

    try:
        # Initialize components
        report_generator_instance = report_generator.UpgradeReportGenerator()
        dependency_updater_instance = dependency_upgrader.DependencyUpdater()
        
        # Setup output directory
        if os.path.exists(new_repo):
            shutil.rmtree(new_repo)
        shutil.copytree(old_repo, new_repo)
        
        print(f"Starting repo upgrade: {old_repo} → {new_repo}")
        
        # Update dependencies
        print("📦 Updating dependencies...")
        dependency_updater_instance.update_requirements_txt(new_repo)
        dependency_updater_instance.update_setup_py(new_repo)
        report_generator_instance.add_dependency_changes(dependency_updater_instance.updated_deps)
        
        # Upgrade Python files
        python_files = []
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
        
        print(f"✅ Upgrade complete! {successful}/{total} files upgraded successfully")
        print(f"📄 Report generated: {report_path}")
        
        return report_path
    finally:
        if previous_project_root is None:
            os.environ.pop("ML_UPGRADER_PROJECT_ROOT", None)
        else:
            os.environ["ML_UPGRADER_PROJECT_ROOT"] = previous_project_root
