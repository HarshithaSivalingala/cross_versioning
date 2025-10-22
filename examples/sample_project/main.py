import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from utils.data_utils import load_processed_data


def summarize_dataset(data_path: Path) -> None:
    features, labels = load_processed_data(data_path)
    print(f"Loaded features shape: {features.shape}")
    print(f"Loaded labels shape: {labels.shape}")

    feature_means = np.mean(features, axis=0).tolist()
    label_distribution = {
        "min": float(np.min(labels)),
        "max": float(np.max(labels)),
        "mean": float(np.mean(labels)),
    }

    summary_path = data_path.with_suffix(".summary.json")
    summary = {
        "feature_means": feature_means,
        "label_stats": label_distribution,
        "observations": int(features.shape[0]),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")


def _run_setup_pipeline(project_root: Path) -> None:
    commands = [
        [sys.executable, "scripts/download_data.py"],
        [sys.executable, "scripts/preprocess_data.py"],
    ]
    for command in commands:
        print(f"Running setup command: {' '.join(command)}")
        result = subprocess.run(command, cwd=project_root)
        if result.returncode != 0:
            raise RuntimeError(f"Setup command failed: {' '.join(command)}")
    print("Setup pipeline completed successfully.")


def main() -> None:
    project_root = Path(__file__).parent
    processed_path = project_root / "data" / "processed" / "dataset.npy"

    if not processed_path.exists():
        print("Processed dataset missing, running setup pipeline...")
        _run_setup_pipeline(project_root)
        if not processed_path.exists():
            raise FileNotFoundError(
                f"Processed dataset still missing at {processed_path} after setup pipeline."
            )

    summarize_dataset(processed_path)

    if os.getenv("ENABLE_LEGACY_TRAINING"):
        # Optional execution path that references legacy ML code.
        from legacy import tf_model, torch_model  # noqa: WPS433

        tf_graph, tf_nodes = tf_model.build_graph()
        print(f"TensorFlow graph contains nodes: {list(tf_nodes)}")

        torch_model.run_training_epoch()


if __name__ == "__main__":
    main()
