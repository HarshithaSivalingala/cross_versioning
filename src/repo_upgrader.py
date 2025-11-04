import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

if __package__ is None or __package__ == "":
    _CURRENT_DIR = Path(__file__).resolve().parent
    _ROOT_DIR = _CURRENT_DIR.parent
    _ROOT_STR = str(_ROOT_DIR)
    if _ROOT_STR not in sys.path:
        sys.path.insert(0, _ROOT_STR)
    import src.agentic_upgrader as agentic_upgrader  # type: ignore
    import src.dependency_analyzer as dependency_analyzer  # type: ignore
    import src.dependency_upgrader as dependency_upgrader  # type: ignore
    import src.report_generator as report_generator  # type: ignore
    import src.validator as validator  # type: ignore
else:
    from . import agentic_upgrader, dependency_analyzer, dependency_upgrader, report_generator, validator


def _ignore_env_dirs(_: str, names: List[str]) -> List[str]:
    ignored = []
    if ".venv" in names:
        ignored.append(".venv")
    if ".ml-upgrader" in names:
        ignored.append(".ml-upgrader")
    return ignored


ProgressCallback = Callable[[str, Optional[float]], None]


def upgrade_repo_with_dependency_awareness(
    old_repo: str,
    new_repo: str,
    dependency_overrides: Optional[Dict[str, str]] = None,
    *,
    verify_runtime_outputs: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """
    Upgrade repository with dependency awareness for better inter-file compatibility.
    
    This is the enhanced version that upgrades files in dependency order,
    so each file knows about the files it imports from. This dramatically
    improves success rate on complex projects with interconnected modules.
    
    Args:
        old_repo: Path to original repository
        new_repo: Path where upgraded repository will be created
        dependency_overrides: Dict of package versions to upgrade to
        verify_runtime_outputs: Whether to validate runtime behavior
        progress_callback: Optional callback for progress updates
    
    Returns:
        Path to generated report
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
                print(f"⚠️  Progress callback error: {exc}")

    try:
        _emit_status("🚀 Preparing repository upgrade...", progress=0, console=False)

        # Run baseline validation if needed
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
                _emit_status("ℹ️  Baseline runtime validation skipped (no runtime command configured).", progress=15)
        else:
            _emit_status(
                "ℹ️  Skipping baseline runtime validation (runtime output comparison disabled).",
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

        _emit_status(f"Starting dependency-aware repo upgrade: {old_repo} → {new_repo}", progress=25)

        # Update dependencies
        _emit_status("📦 Updating dependencies...", progress=30)
        dependency_updater_instance.update_requirements_txt(new_repo)
        dependency_updater_instance.update_setup_py(new_repo)
        report_generator_instance.add_dependency_changes(dependency_updater_instance.updated_deps)
        _emit_status("✅ Dependencies updated.", progress=35, console=False)

        # Analyze dependencies and get upgrade order
        _emit_status("🔍 Analyzing repository structure and dependencies...", progress=40)
        analyzer = dependency_analyzer.DependencyAnalyzer(new_repo)
        analyzer.analyze_repository()
        ordered_files = analyzer.get_upgrade_order()
        
        total_python_files = len(ordered_files)
        
        if not total_python_files:
            _emit_status("ℹ️  No Python files found to upgrade.", progress=80)
        else:
            # Sequential upgrade with dependency context
            _emit_status(
                f"🔄 Upgrading {total_python_files} Python files in dependency order...", 
                progress=45
            )
            _emit_status("   This processes files sequentially to maintain compatibility.", progress=45, console=False)
            
            start_time = time.time()
            upgraded_cache = {}  # Store upgraded file paths for context building
            
            for index, file_path in enumerate(ordered_files):
                try:
                    # Skip special files
                    if '__pycache__' in file_path or '.pyc' in file_path:
                        continue

                    filename = os.path.basename(file_path)
                    rel_path = os.path.relpath(file_path, new_repo)
                    
                    # Skip macOS metadata files
                    if filename.startswith('._'):
                        print(f"ℹ️  Skipping macOS resource fork file: {rel_path}")
                        continue
                    if os.sep + '__MACOSX' + os.sep in file_path or rel_path.startswith('__MACOSX'):
                        print(f"ℹ️  Skipping macOS metadata file: {rel_path}")
                        continue
                    
                    # Get dependencies for this file
                    dependencies = analyzer.get_direct_dependencies(file_path)
                    
                    # Build context from already-upgraded dependencies
                    dependency_context = {}
                    for dep_file in dependencies:
                        if dep_file in upgraded_cache:
                            # Extract interface summary from upgraded file
                            summary = dependency_analyzer.extract_interface_summary(dep_file)
                            if summary:
                                dependency_context[dep_file] = summary
                    
                    # Upgrade file with or without context
                    if dependency_context:
                        print(f"📚 Upgrading {rel_path} with context from {len(dependency_context)} dependencies")
                        result = agentic_upgrader.upgrade_file_with_context(
                            file_path, 
                            file_path, 
                            dependency_context
                        )
                    else:
                        # No dependencies, use regular upgrade
                        result = agentic_upgrader.upgrade_file(file_path, file_path)
                    
                    # Store in cache if successful
                    if result.success:
                        upgraded_cache[file_path] = True
                    
                    report_generator_instance.add_file_result(result)
                    
                    # Update progress
                    progress = 45 + ((index + 1) / total_python_files) * 35
                    _emit_status(
                        f"Progress: {index + 1}/{total_python_files} files processed",
                        progress=progress,
                        console=False
                    )

                except Exception as e:
                    print(f"⚠️  Error upgrading {file_path}: {e}")
                    result = report_generator.FileUpgradeResult(
                        file_path=file_path,
                        success=False,
                        attempts=0,
                        api_changes=[],
                        error=str(e)
                    )
                    report_generator_instance.add_file_result(result)

            elapsed_time = time.time() - start_time
            throughput = total_python_files / elapsed_time if elapsed_time > 0 else 0
            _emit_status(
                f"⏱️  Completed {total_python_files} files in {elapsed_time:.2f}s ({throughput:.2f} files/sec)",
                progress=82
            )

        # Generate report
        report_path = os.path.join(new_repo, "UPGRADE_REPORT.md")
        _emit_status("📄 Generating upgrade report...", progress=85, console=False)
        report_generator_instance.generate_report(report_path)

        successful = len([r for r in report_generator_instance.results if r.success])
        total = len(report_generator_instance.results)

        # Run runtime validation on upgraded project
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
            _emit_status("⚠️  Skipping runtime validation because some files failed validation", progress=95)

        _emit_status(f"✅ Upgrade complete! {successful}/{total} files upgraded successfully", progress=100)
        _emit_status(f"📄 Report generated: {report_path}", progress=100)

        return report_path
        
    finally:
        # Cleanup
        if baseline_temp_dir and os.path.isdir(baseline_temp_dir):
            shutil.rmtree(baseline_temp_dir, ignore_errors=True)
        if previous_project_root is None:
            os.environ.pop("ML_UPGRADER_PROJECT_ROOT", None)
        else:
            os.environ["ML_UPGRADER_PROJECT_ROOT"] = previous_project_root


def upgrade_repo_parallel(
    old_repo: str,
    new_repo: str,
    dependency_overrides: Optional[Dict[str, str]] = None,
    *,
    verify_runtime_outputs: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """
    Original parallel upgrade (fast but may break inter-file dependencies).
    
    This processes files in parallel for speed but doesn't maintain inter-file
    compatibility. Use upgrade_repo_with_dependency_awareness() for better results
    on complex projects.
    
    Returns path to generated report.
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
                print(f"⚠️  Progress callback error: {exc}")

    try:
        _emit_status("🚀 Preparing repository upgrade...", progress=0, console=False)

        # Run baseline validation if needed
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
                _emit_status("ℹ️  Baseline runtime validation skipped (no runtime command configured).", progress=15)
        else:
            _emit_status(
                "ℹ️  Skipping baseline runtime validation (runtime output comparison disabled).",
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

        # Collect all Python files
        python_files: List[str] = []
        for root, _, files in os.walk(new_repo):
            for f in files:
                if f.endswith(".py"):
                    python_files.append(os.path.join(root, f))

        total_python_files = len(python_files)
        
        if not total_python_files:
            _emit_status("ℹ️  No Python files found to upgrade.", progress=80)
        else:
            # Parallel file processing
            max_workers = int(os.getenv("ML_UPGRADER_MAX_WORKERS", "5"))
            _emit_status(f"🔄 Upgrading {total_python_files} Python files with {max_workers} workers...", progress=40)
            
            start_time = time.time()
            completed_count = 0

            def process_single_file(file_path: str) -> Optional[report_generator.FileUpgradeResult]:
                """Process a single Python file and return the result."""
                try:
                    # Skip __pycache__ and generated files
                    if '__pycache__' in file_path or '.pyc' in file_path:
                        return None

                    filename = os.path.basename(file_path)
                    rel_path = os.path.relpath(file_path, new_repo)
                    
                    # Skip macOS metadata files
                    if filename.startswith('._'):
                        print(f"ℹ️  Skipping macOS resource fork file: {rel_path}")
                        return None
                    if os.sep + '__MACOSX' + os.sep in file_path or rel_path.startswith('__MACOSX'):
                        print(f"ℹ️  Skipping macOS metadata file: {rel_path}")
                        return None

                    return agentic_upgrader.upgrade_file(file_path, file_path)

                except Exception as e:
                    print(f"⚠️  Error upgrading {file_path}: {e}")
                    return report_generator.FileUpgradeResult(
                        file_path=file_path,
                        success=False,
                        attempts=0,
                        api_changes=[],
                        error=str(e)
                    )

            # Execute upgrades in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(process_single_file, fp): fp for fp in python_files}
                
                for future in as_completed(future_to_file):
                    result = future.result()
                    if result:
                        report_generator_instance.add_file_result(result)
                    
                    completed_count += 1
                    progress = 40 + (completed_count / total_python_files) * 40
                    file_path = future_to_file[future]
                    rel_path = os.path.relpath(file_path, new_repo)
                    _emit_status(
                        f"Progress: {completed_count}/{total_python_files} files processed",
                        progress=progress,
                        console=False
                    )

            elapsed_time = time.time() - start_time
            throughput = total_python_files / elapsed_time if elapsed_time > 0 else 0
            _emit_status(
                f"⏱️  Completed {total_python_files} files in {elapsed_time:.2f}s ({throughput:.2f} files/sec)",
                progress=82
            )

        # Generate report
        report_path = os.path.join(new_repo, "UPGRADE_REPORT.md")
        _emit_status("📄 Generating upgrade report...", progress=85, console=False)
        report_generator_instance.generate_report(report_path)

        successful = len([r for r in report_generator_instance.results if r.success])
        total = len(report_generator_instance.results)

        # Run runtime validation on upgraded project
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
            _emit_status("⚠️  Skipping runtime validation because some files failed validation", progress=95)

        _emit_status(f"✅ Upgrade complete! {successful}/{total} files upgraded successfully", progress=100)
        _emit_status(f"📄 Report generated: {report_path}", progress=100)

        return report_path
        
    finally:
        # Cleanup
        if baseline_temp_dir and os.path.isdir(baseline_temp_dir):
            shutil.rmtree(baseline_temp_dir, ignore_errors=True)
        if previous_project_root is None:
            os.environ.pop("ML_UPGRADER_PROJECT_ROOT", None)
        else:
            os.environ["ML_UPGRADER_PROJECT_ROOT"] = previous_project_root


def upgrade_repo(
    old_repo: str,
    new_repo: str,
    dependency_overrides: Optional[Dict[str, str]] = None,
    *,
    verify_runtime_outputs: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    use_dependency_order: bool = True,
) -> str:
    """
    Upgrade entire repository with comprehensive reporting.
    
    This is the main entry point for repository upgrades. It can use either
    dependency-aware sequential processing (recommended) or fast parallel
    processing (may break inter-file dependencies).
    
    Args:
        old_repo: Path to original repository
        new_repo: Path where upgraded repo will be created
        dependency_overrides: Dict of package versions to upgrade to
        verify_runtime_outputs: Whether to validate runtime behavior
        progress_callback: Optional callback for progress updates
        use_dependency_order: If True, uses smart dependency-aware upgrade (recommended).
                             If False, uses faster parallel upgrade.
    
    Returns:
        Path to generated report
    """
    if use_dependency_order:
        print("🎯 Using dependency-aware upgrade mode (recommended for complex projects)")
        return upgrade_repo_with_dependency_awareness(
            old_repo, new_repo, dependency_overrides,
            verify_runtime_outputs=verify_runtime_outputs,
            progress_callback=progress_callback
        )
    else:
        print("⚡ Using fast parallel upgrade mode (may break inter-file dependencies)")
        return upgrade_repo_parallel(
            old_repo, new_repo, dependency_overrides,
            verify_runtime_outputs=verify_runtime_outputs,
            progress_callback=progress_callback
        )