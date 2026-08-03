from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from .config import DataConfig

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CLASS_DIRECTORIES = {"Healthy": ("normal", 0), "Brain Tumor": ("tumor", 1)}


def resolve_dataset_root(source: Path) -> Path:
    """Locate the directory that directly contains both expected class folders."""
    source = Path(source).expanduser().resolve()
    candidates = [source, *[path for path in source.rglob("*") if path.is_dir()]]
    for candidate in candidates:
        if all((candidate / folder).is_dir() for folder in CLASS_DIRECTORIES):
            return candidate
    raise FileNotFoundError(
        f"Could not find class folders {list(CLASS_DIRECTORIES)} below {source}."
    )


def _open_grayscale(image_or_path: Image.Image | Path | str) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        return ImageOps.exif_transpose(image_or_path).convert("L")
    with Image.open(image_or_path) as source:
        return ImageOps.exif_transpose(source).convert("L")


def standardize_image(image: Image.Image, image_size: int = 128) -> Image.Image:
    """Pad to a square without distortion, then resize to the model input size."""
    if image_size < 1:
        raise ValueError("image_size must be positive.")
    grayscale = _open_grayscale(image)
    width, height = grayscale.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    padded = Image.new("L", (side, side), color=0)
    padded.paste(grayscale, (left, top))
    return padded.resize((image_size, image_size), Image.Resampling.BILINEAR)


def load_standardized_image(image_path: Path | str, image_size: int = 128) -> Image.Image:
    return standardize_image(_open_grayscale(image_path), image_size=image_size)


def standardized_pixel_hash(image_path: Path | str, image_size: int = 128) -> str:
    pixels = np.asarray(load_standardized_image(image_path, image_size=image_size), dtype=np.uint8)
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def iter_image_paths(dataset_root: Path) -> Iterable[tuple[Path, str, int]]:
    for folder_name, (label_name, label) in CLASS_DIRECTORIES.items():
        class_root = dataset_root / folder_name
        for image_path in sorted(class_root.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_SUFFIXES:
                yield image_path, label_name, label


def audit_dataset(
    source: Path,
    image_size: int = 128,
    excluded_paths: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    dataset_root = resolve_dataset_root(source)
    excluded_paths = excluded_paths or set()
    rows: list[dict[str, object]] = []
    unreadable: list[str] = []

    for image_path, label_name, label in iter_image_paths(dataset_root):
        try:
            with Image.open(image_path) as image:
                image.load()
                original_width, original_height = image.size
                original_format = image.format or image_path.suffix.lstrip(".").upper()
            pixel_hash = standardized_pixel_hash(image_path, image_size=image_size)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            unreadable.append(f"{image_path.relative_to(dataset_root)}: {error}")
            continue

        rows.append(
            {
                "relative_path": image_path.relative_to(dataset_root).as_posix(),
                "label_name": label_name,
                "label": label,
                "original_format": original_format,
                "original_width": original_width,
                "original_height": original_height,
                "pixel_hash": pixel_hash,
            }
        )

    manifest = (
        pd.DataFrame(rows)
        .sort_values(["pixel_hash", "relative_path"])
        .reset_index(drop=True)
    )
    if manifest.empty:
        raise ValueError(f"No readable supported images were found below {dataset_root}.")

    readable_count = len(manifest)
    applied_exclusions = sorted(set(manifest["relative_path"]) & excluded_paths)
    manifest = manifest[~manifest["relative_path"].isin(excluded_paths)].copy()
    if manifest.empty:
        raise ValueError("All readable images were excluded from the dataset.")

    hash_label_counts = manifest.groupby("pixel_hash")["label"].nunique()
    cross_label_duplicate_groups = int((hash_label_counts > 1).sum())
    if cross_label_duplicate_groups:
        raise ValueError(
            f"Found {cross_label_duplicate_groups} pixel-identical groups with conflicting labels."
        )

    group_sizes = manifest.groupby("pixel_hash").size()
    deduplicated = manifest.drop_duplicates("pixel_hash", keep="first").copy()
    summary: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "images_discovered": int(readable_count + len(unreadable)),
        "images_readable": int(readable_count),
        "images_unreadable": int(len(unreadable)),
        "images_excluded": int(len(applied_exclusions)),
        "applied_exclusions": applied_exclusions,
        "unique_analytical_images": int(len(deduplicated)),
        "duplicate_copies_removed": int(len(manifest) - len(deduplicated)),
        "duplicate_groups": int((group_sizes > 1).sum()),
        "cross_label_duplicate_groups": cross_label_duplicate_groups,
        "class_counts_before_deduplication": {
            key: int(value) for key, value in manifest["label_name"].value_counts().items()
        },
        "class_counts_after_deduplication": {
            key: int(value) for key, value in deduplicated["label_name"].value_counts().items()
        },
        "unreadable_examples": unreadable[:10],
    }
    return deduplicated.reset_index(drop=True), summary


def assign_splits(manifest: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    config.validate()
    required = {"relative_path", "label", "label_name", "pixel_hash"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest["pixel_hash"].duplicated().any():
        raise ValueError("assign_splits expects one row per unique analytical image.")

    train, holdout = train_test_split(
        manifest,
        train_size=config.train_fraction,
        random_state=config.seed,
        stratify=manifest["label"],
    )
    validation_share_of_holdout = config.validation_fraction / (
        config.validation_fraction + config.test_fraction
    )
    validation, test = train_test_split(
        holdout,
        train_size=validation_share_of_holdout,
        random_state=config.seed + 1,
        stratify=holdout["label"],
    )

    result = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    ).sort_values(["split", "label", "relative_path"])

    split_hashes = {
        split: set(group["pixel_hash"]) for split, group in result.groupby("split", observed=True)
    }
    assert split_hashes["train"].isdisjoint(split_hashes["validation"])
    assert split_hashes["train"].isdisjoint(split_hashes["test"])
    assert split_hashes["validation"].isdisjoint(split_hashes["test"])
    return result.reset_index(drop=True)


def write_audit_artifacts(
    manifest: pd.DataFrame,
    summary: dict[str, object],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "audit_summary.json"
    manifest.to_csv(manifest_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest_path, summary_path
