from datetime import datetime

import lightgbm as lgb
import numpy as np

from src.data.load_dataset import load_full_dataset
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    labelencode_labels,
    prep_for_model,
)
from src.models.predict_model import score_inputs_cv


def kfold_train(
    nfold: int = 5, num_round: int = 10, num_leaves: int = 31, score: bool = True
) -> dict:
    train_values, train_labels, test_values = load_full_dataset()

    train_values, value_encoders = prep_for_model(train_values)
    train_labels, label_encoder = labelencode_labels(train_labels)

    train_data = lgb.Dataset(
        train_values,
        label=train_labels,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )

    parameter_kwargs = {
        "num_leaves": num_leaves,
        "objective": "multiclass",
        "num_class": 3,
    }
    results = lgb.cv(
        parameter_kwargs, train_data, num_round, nfold=nfold, return_cvbooster=True
    )
    if score:
        scores = score_inputs_cv(results["cvbooster"], train_values, train_labels)
    else:
        scores = None

    return results, scores, value_encoders, label_encoder


def main():
    bst, scores, value_encoders, label_encoder = kfold_train()
    boosters = bst["cvbooster"]
    print(scores)
    best_booster = boosters.boosters[np.argmax(scores)]
    ts = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    best_booster.save_model(f"models/lgbm{ts}.txt")
    breakpoint()


if __name__ == "__main__":
    main()
