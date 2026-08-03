from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DataConfig, TrainConfig
from .preprocessing import assign_splits, audit_dataset, write_audit_artifacts
from .reporting import generate_report_assets
from .training import calibrate_multitask_artifacts, train_experiment


def _audit(arguments: argparse.Namespace) -> None:
    config = DataConfig(image_size=arguments.image_size, seed=arguments.seed)
    exclusions_path = Path(arguments.exclusions)
    excluded_paths: set[str] = set()
    if exclusions_path.is_file():
        import pandas as pd

        exclusions = pd.read_csv(exclusions_path)
        excluded_paths = set(exclusions["relative_path"].astype(str))
    manifest, summary = audit_dataset(
        Path(arguments.source),
        image_size=config.image_size,
        excluded_paths=excluded_paths,
    )
    manifest = assign_splits(manifest, config)
    summary["split_counts"] = {
        key: int(value) for key, value in manifest["split"].value_counts().items()
    }
    manifest_path, summary_path = write_audit_artifacts(
        manifest, summary, Path(arguments.output_dir)
    )
    print(json.dumps(summary, indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


def _train(arguments: argparse.Namespace) -> None:
    config = TrainConfig(
        model_name=arguments.model,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        integrity_loss_weight=arguments.integrity_loss_weight,
        device=arguments.device,
        seed=arguments.seed,
        initial_checkpoint=arguments.initial_checkpoint,
        freeze_tumor_backbone=arguments.freeze_tumor_backbone,
    )
    artifacts = train_experiment(
        manifest_path=Path(arguments.manifest),
        data_root=Path(arguments.data_root),
        artifact_dir=Path(arguments.artifact_dir),
        config=config,
        image_size=arguments.image_size,
    )
    print(json.dumps({key: str(value) for key, value in artifacts.__dict__.items()}, indent=2))


def _calibrate(arguments: argparse.Namespace) -> None:
    thresholds = calibrate_multitask_artifacts(
        manifest_path=Path(arguments.manifest),
        data_root=Path(arguments.data_root),
        artifact_dir=Path(arguments.artifact_dir),
    )
    print(json.dumps(thresholds, indent=2))


def _report(arguments: argparse.Namespace) -> None:
    paths = generate_report_assets(
        artifact_dir=Path(arguments.artifact_dir),
        report_dir=Path(arguments.report_dir),
        manifest_path=Path(arguments.manifest),
        data_root=Path(arguments.data_root),
    )
    print(json.dumps([str(path) for path in paths], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mri-integrity",
        description="Audit data and run MRI Integrity Lab experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit", help="Decode, deduplicate, and split the source dataset."
    )
    audit.add_argument("--source", required=True)
    audit.add_argument("--output-dir", default="data/processed")
    audit.add_argument("--exclusions", default="data/exclusions.csv")
    audit.add_argument("--image-size", type=int, default=128)
    audit.add_argument("--seed", type=int, default=5910)
    audit.set_defaults(handler=_audit)

    train = subparsers.add_parser("train", help="Train and evaluate one model experiment.")
    train.add_argument("--model", choices=["baseline", "improved", "multitask"], required=True)
    train.add_argument("--manifest", default="data/processed/manifest.csv")
    train.add_argument("--data-root", required=True)
    train.add_argument("--artifact-dir", default="artifacts")
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--integrity-loss-weight", type=float, default=0.5)
    train.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    train.add_argument("--image-size", type=int, default=128)
    train.add_argument("--seed", type=int, default=5910)
    train.add_argument("--initial-checkpoint")
    train.add_argument("--freeze-tumor-backbone", action="store_true")
    train.set_defaults(handler=_train)

    calibrate = subparsers.add_parser(
        "calibrate", help="Calibrate a trained integrity head on validation data."
    )
    calibrate.add_argument("--manifest", default="data/processed/manifest.csv")
    calibrate.add_argument("--data-root", required=True)
    calibrate.add_argument("--artifact-dir", default="artifacts")
    calibrate.set_defaults(handler=_calibrate)

    report = subparsers.add_parser("report", help="Generate factual report assets and figures.")
    report.add_argument("--manifest", default="data/processed/manifest.csv")
    report.add_argument("--data-root", required=True)
    report.add_argument("--artifact-dir", default="artifacts")
    report.add_argument("--report-dir", default="reports")
    report.set_defaults(handler=_report)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
