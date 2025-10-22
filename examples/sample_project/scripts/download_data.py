import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import write_csv  # noqa: E402
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"


def generate_rows(num_rows: int = 64, seed: int = 13):
    random.seed(seed)
    yield ["f1", "f2", "f3", "f4", "label"]
    for _ in range(num_rows):
        features = [round(random.random(), 6) for _ in range(4)]
        label = round(sum(features) / len(features), 6)
        yield [*features, label]


def main():
    rows = list(generate_rows())
    write_csv(RAW_DATA_PATH, rows)
    print(f"Wrote synthetic dataset to {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
