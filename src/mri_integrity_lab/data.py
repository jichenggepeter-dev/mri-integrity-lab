from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from .preprocessing import load_standardized_image, resolve_dataset_root
from .tampering import TamperType, apply_tampering


def attach_image_paths(manifest: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    resolved_root = resolve_dataset_root(data_root)
    result = manifest.copy()
    result["image_path"] = result["relative_path"].map(lambda value: resolved_root / value)
    missing = [str(path) for path in result["image_path"] if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Manifest references {len(missing)} missing images; first: {missing[0]}"
        )
    return result


def compute_normalization_stats(
    frame: pd.DataFrame,
    image_size: int = 128,
) -> tuple[float, float]:
    pixel_sum = 0.0
    squared_sum = 0.0
    pixel_count = 0
    for image_path in frame["image_path"]:
        pixels = np.asarray(
            load_standardized_image(image_path, image_size=image_size), dtype=np.float32
        ) / 255.0
        pixel_sum += float(pixels.sum())
        squared_sum += float(np.square(pixels).sum())
        pixel_count += int(pixels.size)
    mean = pixel_sum / pixel_count
    variance = max(squared_sum / pixel_count - mean**2, 1e-8)
    return float(mean), float(np.sqrt(variance))


def _augment(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    angle = random.uniform(-8.0, 8.0)
    return image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)


def _to_tensor(image: Image.Image | np.ndarray, mean: float, std: float) -> torch.Tensor:
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - mean) / max(std, 1e-6)
    return torch.from_numpy(pixels.copy()).unsqueeze(0)


class TumorDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        mean: float,
        std: float,
        image_size: int = 128,
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.mean = float(mean)
        self.std = float(std)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image = load_standardized_image(row["image_path"], image_size=self.image_size)
        if self.augment:
            image = _augment(image)
        return {
            "image": _to_tensor(image, self.mean, self.std),
            "tumor_label": torch.tensor(int(row["label"]), dtype=torch.long),
            "integrity_label": torch.tensor(0, dtype=torch.long),
            "sample_id": str(row["relative_path"]),
            "variant": "clean",
            "tamper_type": "clean",
        }


class MultiTaskDataset(Dataset[dict[str, object]]):
    """Yield one clean and one deterministic manipulated version of each original image."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        mean: float,
        std: float,
        image_size: int = 128,
        augment: bool = False,
        seed: int = 5910,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.mean = float(mean)
        self.std = float(std)
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self.tamper_types = tuple(TamperType)

    def __len__(self) -> int:
        return len(self.frame) * 2

    def __getitem__(self, index: int) -> dict[str, object]:
        base_index = index // 2
        is_manipulated = index % 2 == 1
        row = self.frame.iloc[base_index]
        image = load_standardized_image(row["image_path"], image_size=self.image_size)
        if self.augment:
            image = _augment(image)

        tamper_type = "clean"
        if is_manipulated:
            selected = self.tamper_types[base_index % len(self.tamper_types)]
            tamper_seed = self.seed + base_index * 1009
            result = apply_tampering(image, selected, seed=tamper_seed)
            model_image: Image.Image | np.ndarray = result.image
            tamper_type = selected.value
        else:
            model_image = image

        return {
            "image": _to_tensor(model_image, self.mean, self.std),
            "tumor_label": torch.tensor(int(row["label"]), dtype=torch.long),
            "integrity_label": torch.tensor(int(is_manipulated), dtype=torch.long),
            "sample_id": str(row["relative_path"]),
            "variant": "manipulated" if is_manipulated else "clean",
            "tamper_type": tamper_type,
        }


def make_loader(
    dataset: Dataset[dict[str, object]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )
