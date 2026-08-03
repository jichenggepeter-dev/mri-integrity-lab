from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .preprocessing import load_standardized_image, resolve_dataset_root
from .tampering import TamperType, apply_tampering

COLORS = {
    "teal": "#006D67",
    "charcoal": "#263238",
    "amber": "#C27A17",
    "red": "#A33B32",
    "gray": "#B8C3C7",
}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _model_comparison(artifact_dir: Path, output_dir: Path) -> None:
    names = ["Baseline", "Improved", "Multi-task"]
    metrics = [
        _load_json(artifact_dir / "baseline_metrics.json"),
        _load_json(artifact_dir / "improved_metrics.json"),
        _load_json(artifact_dir / "multitask_metrics.json"),
    ]
    accuracy = [float(item["test_tumor_clean"]["accuracy"]) for item in metrics]  # type: ignore[index]
    roc_auc = [float(item["test_tumor_clean"]["roc_auc"]) for item in metrics]  # type: ignore[index]

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(len(names))
    width = 0.34
    axis.bar(
        positions - width / 2,
        accuracy,
        width,
        label="Accuracy",
        color=COLORS["teal"],
    )
    axis.bar(
        positions + width / 2,
        roc_auc,
        width,
        label="ROC-AUC",
        color=COLORS["amber"],
    )
    axis.set_ylim(0.0, 1.01)
    axis.set_ylabel("Held-out test score")
    axis.set_xticks(positions, names)
    axis.set_title("Tumor classification model comparison")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, output_dir / "model_comparison.png")


def _training_curves(artifact_dir: Path, output_dir: Path) -> None:
    experiments = [
        ("Baseline", "baseline_history.csv", COLORS["gray"]),
        ("Improved", "improved_history.csv", COLORS["teal"]),
        ("Final multi-task", "multitask_history.csv", COLORS["red"]),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for name, filename, color in experiments:
        history = pd.read_csv(artifact_dir / filename)
        axes[0].plot(history["epoch"], history["train_loss"], color=color, label=name)
        axes[1].plot(
            history["epoch"], history["validation_loss"], color=color, label=name
        )
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Recorded objective")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    _save_figure(figure, output_dir / "training_curves.png")


def _confusion_matrices(artifact_dir: Path, output_dir: Path) -> None:
    metrics = _load_json(artifact_dir / "multitask_metrics.json")
    matrices = [
        (
            "Tumor classification",
            metrics["test_tumor_clean"]["confusion_matrix"],  # type: ignore[index]
            ["Normal", "Tumor"],
        ),
        (
            "Integrity screening",
            metrics["test_integrity"]["confusion_matrix"],  # type: ignore[index]
            ["Clean", "Manipulated"],
        ),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for axis, (title, values, labels) in zip(axes, matrices, strict=True):
        matrix = np.asarray(values)
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                    fontsize=12,
                    fontweight="bold",
                )
        axis.set_title(title)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1], labels, rotation=10)
        axis.set_yticks([0, 1], labels)
    _save_figure(figure, output_dir / "confusion_matrices.png")


def _integrity_iterations(artifact_dir: Path, output_dir: Path) -> None:
    versions = [
        ("Global pool\nunpaired", artifact_dir / "v1_global_pool" / "multitask_metrics.json"),
        (
            "Residual branch\nunpaired",
            artifact_dir / "v2_residual_unpaired" / "multitask_metrics.json",
        ),
        ("Residual branch\npaired", artifact_dir / "multitask_metrics.json"),
    ]
    accuracy = []
    roc_auc = []
    for _, path in versions:
        values = _load_json(path)["test_integrity"]
        accuracy.append(float(values["accuracy"]))  # type: ignore[index]
        roc_auc.append(float(values["roc_auc"]))  # type: ignore[index]

    figure, axis = plt.subplots(figsize=(7.4, 4.3))
    positions = np.arange(len(versions))
    width = 0.34
    axis.bar(
        positions - width / 2,
        accuracy,
        width,
        label="Accuracy",
        color=COLORS["charcoal"],
    )
    axis.bar(
        positions + width / 2,
        roc_auc,
        width,
        label="ROC-AUC",
        color=COLORS["amber"],
    )
    axis.set_ylim(0.45, 0.92)
    axis.set_xticks(positions, [item[0] for item in versions])
    axis.set_ylabel("Held-out test score")
    axis.set_title("Integrity detector iteration")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, output_dir / "integrity_iterations.png")


def _robustness(artifact_dir: Path, output_dir: Path) -> None:
    metrics = _load_json(artifact_dir / "multitask_metrics.json")
    frame = pd.DataFrame(metrics["test_robustness"]["by_tamper_type"])  # type: ignore[index]
    labels = frame["tamper_type"].str.replace("_", " ").str.title()
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].bar(labels, frame["mean_absolute_shift"], color=COLORS["teal"])
    axes[0].set_title("Tumor probability shift")
    axes[0].set_ylabel("Mean absolute shift")
    axes[1].bar(labels, frame["flip_rate"], color=COLORS["red"])
    axes[1].set_title("Diagnosis flip rate")
    axes[1].set_ylabel("Fraction of paired images")
    for axis in axes:
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, output_dir / "robustness_by_tamper.png")


def _tamper_examples(
    artifact_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    data_root: Path,
) -> None:
    del artifact_dir
    manifest = pd.read_csv(manifest_path)
    example = manifest[(manifest["split"] == "test") & (manifest["label"] == 1)].iloc[0]
    root = resolve_dataset_root(data_root)
    image = load_standardized_image(root / example["relative_path"], image_size=128)
    panels = [("Standardized input", np.asarray(image))]
    for index, tamper_type in enumerate(TamperType):
        result = apply_tampering(image, tamper_type, seed=5910 + index)
        panels.append((tamper_type.value.replace("_", " ").title(), result.image))
    figure, axes = plt.subplots(1, 4, figsize=(11.0, 3.0))
    for axis, (title, pixels) in zip(axes, panels, strict=True):
        axis.imshow(pixels, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    _save_figure(figure, output_dir / "tamper_examples.png")


def _write_evidence_pack(artifact_dir: Path, report_dir: Path) -> None:
    baseline = _load_json(artifact_dir / "baseline_metrics.json")
    improved = _load_json(artifact_dir / "improved_metrics.json")
    final = _load_json(artifact_dir / "multitask_metrics.json")
    audit = _load_json(Path("data/processed/audit_summary.json"))
    rows = [
        {
            "model": "baseline",
            **baseline["test_tumor_clean"],  # type: ignore[misc]
        },
        {
            "model": "improved",
            **improved["test_tumor_clean"],  # type: ignore[misc]
        },
        {
            "model": "multitask_clean",
            **final["test_tumor_clean"],  # type: ignore[misc]
        },
        {
            "model": "multitask_integrity",
            **final["test_integrity"],  # type: ignore[misc]
        },
    ]
    pd.DataFrame(rows).to_csv(report_dir / "metrics_summary.csv", index=False)

    robustness = final["test_robustness"]  # type: ignore[assignment]
    content = f"""# Evidence Pack (factual notes only)

This file contains measured facts and source pointers. It is not a report draft. The student must
write the submitted 3-4 page report independently.

## Identity

- Student: Jicheng Ge
- UNI: jg5013

## Data audit

- Public source: Preet Viradiya, Brain Tumor Dataset on Kaggle.
- Discovered/readable files: {audit['images_readable']}
- Explicitly excluded non-brain image: {audit['images_excluded']}
- Duplicate copies removed: {audit['duplicate_copies_removed']}
- Deduplicated analytical images: {audit['unique_analytical_images']}
- Split counts: {audit['split_counts']}

## Held-out test facts

- Baseline tumor accuracy: {baseline['test_tumor_clean']['accuracy']:.4f}
- Improved tumor accuracy: {improved['test_tumor_clean']['accuracy']:.4f}
- Final clean-image tumor accuracy: {final['test_tumor_clean']['accuracy']:.4f}
- Final tumor ROC-AUC: {final['test_tumor_clean']['roc_auc']:.4f}
- Integrity accuracy: {final['test_integrity']['accuracy']:.4f}
- Integrity ROC-AUC: {final['test_integrity']['roc_auc']:.4f}
- Paired diagnosis flip rate: {robustness['prediction_flip_rate']:.4f}
- Mean absolute tumor-probability shift: {robustness['mean_absolute_probability_shift']:.4f}

## Interpretation constraints

- The labels are image-level binary classes, not clinical diagnoses or tumor subtypes.
- Integrity labels are synthetic and cover copy-move, local occlusion, and intensity/noise only.
- The source contains no patient identifiers, so deduplication is image-hash based rather than
  patient-group based.
- The model is not externally validated and is not a medical device or forensic detector.
- Do not claim that a high integrity score proves malicious alteration.

## Source links

- Dataset: https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset
- Dataset DOI: https://doi.org/10.3740/KAGGLE/DSV/1183165
- PyTorch: https://pytorch.org/
- Streamlit: https://streamlit.io/
"""
    (report_dir / "evidence_pack.md").write_text(content, encoding="utf-8")


def generate_report_assets(
    *,
    artifact_dir: Path,
    report_dir: Path,
    manifest_path: Path,
    data_root: Path,
) -> list[Path]:
    output_dir = report_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    _model_comparison(artifact_dir, output_dir)
    _training_curves(artifact_dir, output_dir)
    _confusion_matrices(artifact_dir, output_dir)
    _integrity_iterations(artifact_dir, output_dir)
    _robustness(artifact_dir, output_dir)
    _tamper_examples(artifact_dir, output_dir, manifest_path, data_root)
    _write_evidence_pack(artifact_dir, report_dir)
    return sorted(output_dir.glob("*.png"))
