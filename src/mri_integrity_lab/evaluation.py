from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics: dict[str, object] = {
        "count": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["pr_auc"] = float(average_precision_score(labels, probabilities))
    return metrics


def robustness_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    required = {"sample_id", "variant", "tumor_probability", "tamper_type"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")

    clean = predictions[predictions["variant"] == "clean"].set_index("sample_id")
    manipulated = predictions[predictions["variant"] == "manipulated"].set_index("sample_id")
    paired = clean[["tumor_probability"]].join(
        manipulated[["tumor_probability", "tamper_type"]],
        how="inner",
        lsuffix="_clean",
        rsuffix="_manipulated",
    )
    if paired.empty:
        raise ValueError("No clean/manipulated prediction pairs were available.")
    paired["probability_shift"] = (
        paired["tumor_probability_manipulated"] - paired["tumor_probability_clean"]
    )
    paired["prediction_flip"] = (
        paired["tumor_probability_clean"] >= 0.5
    ) != (paired["tumor_probability_manipulated"] >= 0.5)

    by_type = (
        paired.groupby("tamper_type")
        .agg(
            count=("probability_shift", "size"),
            mean_absolute_shift=("probability_shift", lambda values: float(np.abs(values).mean())),
            flip_rate=("prediction_flip", "mean"),
        )
        .reset_index()
    )
    return {
        "paired_count": int(len(paired)),
        "mean_absolute_probability_shift": float(paired["probability_shift"].abs().mean()),
        "median_absolute_probability_shift": float(paired["probability_shift"].abs().median()),
        "prediction_flip_rate": float(paired["prediction_flip"].mean()),
        "by_tamper_type": by_type.to_dict(orient="records"),
    }


def write_metrics(metrics: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

