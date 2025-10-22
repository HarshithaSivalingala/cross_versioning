import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.npy"
METADATA_PATH = PROCESSED_DATA_PATH.with_suffix(".meta.json")


def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Expected raw data at {RAW_DATA_PATH}")

    raw = np.genfromtxt(RAW_DATA_PATH, delimiter=",", skip_header=1)
    features = raw[:, :4]
    labels = raw[:, 4:]

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        PROCESSED_DATA_PATH,
        {"features": features, "labels": labels},
        allow_pickle=True,
    )

    metadata = {
        "rows": int(raw.shape[0]),
        "feature_columns": 4,
        "label_columns": 1,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved processed dataset to {PROCESSED_DATA_PATH}")
    print(f"Wrote metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
