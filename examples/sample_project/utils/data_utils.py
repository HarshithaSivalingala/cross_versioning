import csv
from pathlib import Path
from typing import Tuple

import numpy as np


def _ensure_array(data):
    if isinstance(data, list):
        return np.array(data)
    return data


def load_processed_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing processed data file at {data_path}")

    loaded = np.load(data_path, allow_pickle=True)
    features = _ensure_array(loaded.item().get("features"))
    labels = _ensure_array(loaded.item().get("labels"))

    try:
        mean_value = np.asscalar(np.mean(features))
    except AttributeError:
        mean_value = np.mean(features).item()

    try:
        cast_features = features.astype(np.float)
    except AttributeError:
        cast_features = features.astype(float)

    print(f"Feature global mean: {mean_value:.4f}")
    return cast_features, labels


def write_csv(path: Path, rows):
    if not isinstance(path, Path):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)
