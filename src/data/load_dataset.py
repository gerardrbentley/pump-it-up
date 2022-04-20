import functools
from pathlib import Path

import pandas as pd


@functools.lru_cache()
def load_full_dataset():
    processed_dir = Path("data") / "processed"
    train_values = pd.read_csv(str(processed_dir / "train_values.csv"), index_col="id")
    train_labels = pd.read_csv(str(processed_dir / "train_labels.csv"), index_col="id")
    test_values = pd.read_csv(str(processed_dir / "test_values.csv"), index_col="id")
    return train_values, train_labels, test_values
