from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn

from .config import TrainConfig
from .data import (
    MultiTaskDataset,
    TumorDataset,
    attach_image_paths,
    compute_normalization_stats,
    make_loader,
)
from .evaluation import binary_metrics, robustness_metrics, write_metrics
from .models import MultiTaskCNN, MultiTaskOutput, build_model, parameter_count


@dataclass(frozen=True)
class TrainArtifacts:
    checkpoint_path: Path
    history_path: Path
    metrics_path: Path
    predictions_path: Path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if requested not in {"cpu", "mps"}:
        raise ValueError("This project supports explicit cpu or mps execution.")
    return torch.device(requested)


def _initialize_multitask_model(model: MultiTaskCNN, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "improved":
        raise ValueError("Multitask initialization requires an improved-model checkpoint.")

    initialized_state = model.state_dict()
    for key, value in checkpoint["state_dict"].items():
        target_key = key
        if key.startswith("classifier."):
            target_key = key.replace("classifier.", "tumor_head.", 1)
        if target_key in initialized_state:
            initialized_state[target_key] = value
    model.load_state_dict(initialized_state)


def _class_weights(labels: pd.Series, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=2)
    weights = len(labels) / (2.0 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _single_epoch(
    model: nn.Module,
    loader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["tumor_label"].to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = loss_function(logits, targets)
            if is_training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(images)
        labels.extend(targets.detach().cpu().tolist())
        predictions.extend(logits.argmax(1).detach().cpu().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "tumor_accuracy": float(accuracy_score(labels, predictions)),
    }


def _multitask_epoch(
    model: MultiTaskCNN,
    loader,
    tumor_loss_function: nn.Module,
    integrity_loss_function: nn.Module,
    config: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals = {"loss": 0.0, "tumor_loss": 0.0, "integrity_loss": 0.0}
    tumor_labels: list[int] = []
    tumor_predictions: list[int] = []
    integrity_labels: list[int] = []
    integrity_predictions: list[int] = []

    for batch in loader:
        images = batch["image"].to(device)
        tumor_targets = batch["tumor_label"].to(device)
        integrity_targets = batch["integrity_label"].to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            output: MultiTaskOutput = model(images)
            tumor_loss = tumor_loss_function(output.tumor_logits, tumor_targets)
            integrity_loss = integrity_loss_function(output.integrity_logits, integrity_targets)
            loss = tumor_loss + config.integrity_loss_weight * integrity_loss
            if is_training:
                loss.backward()
                optimizer.step()
        batch_size = len(images)
        totals["loss"] += float(loss.item()) * batch_size
        totals["tumor_loss"] += float(tumor_loss.item()) * batch_size
        totals["integrity_loss"] += float(integrity_loss.item()) * batch_size
        tumor_labels.extend(tumor_targets.detach().cpu().tolist())
        tumor_predictions.extend(output.tumor_logits.argmax(1).detach().cpu().tolist())
        integrity_labels.extend(integrity_targets.detach().cpu().tolist())
        integrity_predictions.extend(output.integrity_logits.argmax(1).detach().cpu().tolist())

    return {
        key: value / len(loader.dataset) for key, value in totals.items()
    } | {
        "tumor_accuracy": float(accuracy_score(tumor_labels, tumor_predictions)),
        "integrity_accuracy": float(accuracy_score(integrity_labels, integrity_predictions)),
    }


def _predict(model: nn.Module, loader, device: torch.device, model_name: str) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            if model_name == "multitask":
                output: MultiTaskOutput = model(images)
                tumor_probabilities = torch.softmax(output.tumor_logits, 1)[:, 1]
                integrity_probabilities = torch.softmax(output.integrity_logits, 1)[:, 1]
            else:
                logits = model(images)
                tumor_probabilities = torch.softmax(logits, 1)[:, 1]
                integrity_probabilities = torch.zeros_like(tumor_probabilities)

            for index in range(len(images)):
                rows.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "variant": batch["variant"][index],
                        "tamper_type": batch["tamper_type"][index],
                        "tumor_label": int(batch["tumor_label"][index]),
                        "integrity_label": int(batch["integrity_label"][index]),
                        "tumor_probability": float(tumor_probabilities[index].cpu()),
                        "integrity_probability": float(integrity_probabilities[index].cpu()),
                    }
                )
    return pd.DataFrame(rows)


def train_experiment(
    *,
    manifest_path: Path,
    data_root: Path,
    artifact_dir: Path,
    config: TrainConfig,
    image_size: int = 128,
) -> TrainArtifacts:
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest = attach_image_paths(pd.read_csv(manifest_path), data_root)
    train_frame = manifest[manifest["split"] == "train"].copy()
    validation_frame = manifest[manifest["split"] == "validation"].copy()
    test_frame = manifest[manifest["split"] == "test"].copy()
    if min(len(train_frame), len(validation_frame), len(test_frame)) == 0:
        raise ValueError("Manifest must contain non-empty train, validation, and test splits.")

    mean, std = compute_normalization_stats(train_frame, image_size=image_size)
    dataset_class = MultiTaskDataset if config.model_name == "multitask" else TumorDataset
    common = {"mean": mean, "std": std, "image_size": image_size}
    if config.model_name == "multitask":
        train_dataset = dataset_class(
            train_frame,
            **common,
            augment=True,
            seed=config.seed,
            paired=False,  # type: ignore[arg-type]
        )
        validation_dataset = dataset_class(
            validation_frame,
            **common,
            augment=False,
            seed=config.seed,
            paired=True,  # type: ignore[arg-type]
        )
        test_dataset = dataset_class(
            test_frame,
            **common,
            augment=False,
            seed=config.seed,
            paired=True,  # type: ignore[arg-type]
        )
    else:
        train_dataset = dataset_class(train_frame, **common, augment=True)
        validation_dataset = dataset_class(validation_frame, **common, augment=False)
        test_dataset = dataset_class(test_frame, **common, augment=False)

    train_loader = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    model = build_model(config.model_name, dropout=config.dropout).to(device)
    if config.initial_checkpoint:
        _initialize_multitask_model(model, Path(config.initial_checkpoint))  # type: ignore[arg-type]
    tumor_loss = nn.CrossEntropyLoss(weight=_class_weights(train_frame["label"], device))
    integrity_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    started_at = time.time()
    for epoch in range(1, config.epochs + 1):
        if config.model_name == "multitask":
            train_result = _multitask_epoch(
                model, train_loader, tumor_loss, integrity_loss, config, device, optimizer  # type: ignore[arg-type]
            )
            validation_result = _multitask_epoch(
                model, validation_loader, tumor_loss, integrity_loss, config, device  # type: ignore[arg-type]
            )
        else:
            train_result = _single_epoch(model, train_loader, tumor_loss, device, optimizer)
            validation_result = _single_epoch(model, validation_loader, tumor_loss, device)
        scheduler.step(validation_result["loss"])
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_result.items()})
        row.update({f"validation_{key}": value for key, value in validation_result.items()})
        row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train {train_result['loss']:.4f} | "
            f"validation {validation_result['loss']:.4f}",
            flush=True,
        )
        if validation_result["loss"] < best_loss - 1e-4:
            best_loss = validation_result["loss"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Training completed without a checkpoint.")
    model.load_state_dict(best_state)

    predictions = _predict(model, test_loader, device, config.model_name)
    clean_predictions = predictions[predictions["variant"] == "clean"]
    metrics: dict[str, object] = {
        "model_name": config.model_name,
        "parameter_count": parameter_count(model),
        "device": str(device),
        "image_size": image_size,
        "normalization": {"mean": mean, "std": std},
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "training_seconds": time.time() - started_at,
        "test_tumor_clean": binary_metrics(
            clean_predictions["tumor_label"].to_numpy(),
            clean_predictions["tumor_probability"].to_numpy(),
        ),
    }
    if config.model_name == "multitask":
        metrics["test_tumor_all_inputs"] = binary_metrics(
            predictions["tumor_label"].to_numpy(),
            predictions["tumor_probability"].to_numpy(),
        )
        metrics["test_integrity"] = binary_metrics(
            predictions["integrity_label"].to_numpy(),
            predictions["integrity_probability"].to_numpy(),
        )
        metrics["test_robustness"] = robustness_metrics(predictions)

    checkpoint_path = artifact_dir / f"{config.model_name}.pt"
    history_path = artifact_dir / f"{config.model_name}_history.csv"
    metrics_path = artifact_dir / f"{config.model_name}_metrics.json"
    predictions_path = artifact_dir / f"{config.model_name}_test_predictions.csv"
    torch.save(
        {
            "model_name": config.model_name,
            "state_dict": model.state_dict(),
            "train_config": asdict(config),
            "image_size": image_size,
            "normalization": {"mean": mean, "std": std},
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(history_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    write_metrics(metrics, metrics_path)
    (artifact_dir / f"{config.model_name}_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    return TrainArtifacts(checkpoint_path, history_path, metrics_path, predictions_path)
