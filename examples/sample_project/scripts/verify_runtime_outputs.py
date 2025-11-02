import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.runtime_validation import perform_project_runtime_validation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / ".ml-upgrader" / "runtime_outputs"
BASELINE_DIR = OUTPUT_ROOT / "sample_baseline"
CHECK_DIR = OUTPUT_ROOT / "sample_check"


def _cleanup_previous_runs() -> None:
    for path in [BASELINE_DIR, CHECK_DIR]:
        if path.exists():
            shutil.rmtree(path)


def _run_baseline() -> None:
    print(f"Running baseline runtime validation for {PROJECT_ROOT}...")
    ok, error = perform_project_runtime_validation(
        str(PROJECT_ROOT),
        output_capture_dir=str(BASELINE_DIR),
    )
    if not ok:
        raise SystemExit(f"Baseline runtime validation failed: {error}")
    if not BASELINE_DIR.is_dir():
        print("ℹ️ No runtime command configured; nothing to compare.")
        raise SystemExit(0)
    print(f"Captured baseline outputs in {BASELINE_DIR}")


def _run_comparison() -> None:
    print("Re-running runtime validation and comparing against the baseline...")
    ok, error = perform_project_runtime_validation(
        str(PROJECT_ROOT),
        output_capture_dir=str(CHECK_DIR),
        compare_with_dir=str(BASELINE_DIR),
    )
    if not ok:
        raise SystemExit(f"Runtime outputs diverged from baseline: {error}")
    print(f"Outputs match! Comparison results saved in {CHECK_DIR}")


def main() -> None:
    if not os.environ.get("PYTHONPATH"):
        os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
    else:
        os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"

    _cleanup_previous_runs()
    _run_baseline()
    _run_comparison()


if __name__ == "__main__":
    main()
