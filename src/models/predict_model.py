from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.load_dataset import load_full_dataset
from src.features.build_features import labelencode_labels, prep_for_model


def classification_rate(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sum(y == y_pred) / len(y)


def make_prediction_cv(boosters, input_data: pd.DataFrame) -> list:
    preds = boosters.predict(input_data)
    return [pd.DataFrame(prediction_set).idxmax(axis=1) for prediction_set in preds]


def score_inputs_cv(boosters, input_data, labels: pd.DataFrame) -> np.ndarray:
    preds = make_prediction_cv(boosters, input_data)
    return np.asarray(
        [
            classification_rate(labels["status_group"].values, pred_labels.values)
            for pred_labels in preds
        ]
    )


def make_prediction(booster, input_data: pd.DataFrame) -> pd.DataFrame:
    preds = booster.predict(input_data)
    return pd.DataFrame(preds).idxmax(axis=1)


def score_inputs(booster, input_data, labels: pd.DataFrame) -> np.ndarray:
    preds = make_prediction(booster, input_data)
    return classification_rate(labels["status_group"].values, preds.values)


def main():
    models = Path("models").glob("*")
    latest_path = max(models, key=lambda p: p.stat().st_ctime)
    bst = lgb.Booster(model_file=str(latest_path))
    train_values, train_labels, test_values = load_full_dataset()
    test_values, encoders = prep_for_model(test_values)
    preds = make_prediction(bst, test_values)
    t, label_encoder = labelencode_labels(train_labels)
    decoded_preds = pd.DataFrame(
        label_encoder.inverse_transform(preds),
        columns=["status_group"],
        index=test_values.index,
    )
    decoded_preds.to_csv(f"reports/submission_{latest_path.stem}.csv")
    breakpoint()


if __name__ == "__main__":
    main()
