from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from PIL import Image


class TamperType(StrEnum):
    COPY_MOVE = "copy_move"
    OCCLUSION = "occlusion"
    INTENSITY_NOISE = "intensity_noise"


@dataclass(frozen=True)
class TamperResult:
    image: np.ndarray
    mask: np.ndarray
    tamper_type: TamperType
    severity: float
    metadata: dict[str, int | float | str]


def _as_grayscale_array(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    if pixels.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image.")
    if min(pixels.shape) < 16:
        raise ValueError("Image is too small for controlled synthetic manipulation.")
    return pixels.copy()


def _random_box(
    rng: np.random.Generator,
    height: int,
    width: int,
    min_fraction: float = 0.14,
    max_fraction: float = 0.28,
) -> tuple[int, int, int, int]:
    box_height = int(
        rng.integers(max(6, int(height * min_fraction)), int(height * max_fraction) + 1)
    )
    box_width = int(
        rng.integers(max(6, int(width * min_fraction)), int(width * max_fraction) + 1)
    )
    top = int(rng.integers(0, height - box_height + 1))
    left = int(rng.integers(0, width - box_width + 1))
    return top, left, box_height, box_width


def apply_tampering(
    image: Image.Image,
    tamper_type: TamperType | str,
    seed: int,
) -> TamperResult:
    tamper_type = TamperType(tamper_type)
    rng = np.random.default_rng(seed)
    original = _as_grayscale_array(image)
    manipulated = original.copy()
    mask = np.zeros_like(original, dtype=np.uint8)
    height, width = original.shape
    top, left, box_height, box_width = _random_box(rng, height, width)
    severity: float
    metadata: dict[str, int | float | str] = {
        "top": top,
        "left": left,
        "height": box_height,
        "width": box_width,
    }

    if tamper_type is TamperType.COPY_MOVE:
        source_top, source_left, _, _ = _random_box(
            rng,
            height,
            width,
            min_fraction=box_height / height,
            max_fraction=box_height / height,
        )
        source_top = min(source_top, height - box_height)
        source_left = min(source_left, width - box_width)
        patch = original[
            source_top : source_top + box_height,
            source_left : source_left + box_width,
        ].astype(np.float32)
        if patch.shape != (box_height, box_width):
            patch = np.asarray(
                Image.fromarray(patch.astype(np.uint8)).resize(
                    (box_width, box_height), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
        gain = float(rng.choice([rng.uniform(0.82, 0.94), rng.uniform(1.06, 1.18)]))
        patch = np.clip(patch * gain, 0, 255).astype(np.uint8)
        manipulated[top : top + box_height, left : left + box_width] = patch
        mask[top : top + box_height, left : left + box_width] = 1
        severity = min(1.0, (box_height * box_width) / (height * width) * 4.0)
        metadata.update(
            {
                "source_top": source_top,
                "source_left": source_left,
                "gain": gain,
            }
        )

    elif tamper_type is TamperType.OCCLUSION:
        fill_value = int(rng.choice([0, 255, int(np.median(original))]))
        manipulated[top : top + box_height, left : left + box_width] = fill_value
        mask[top : top + box_height, left : left + box_width] = 1
        severity = min(1.0, (box_height * box_width) / (height * width) * 4.0)
        metadata["fill_value"] = fill_value

    else:
        region = manipulated[top : top + box_height, left : left + box_width].astype(np.float32)
        contrast = float(rng.choice([rng.uniform(0.55, 0.80), rng.uniform(1.25, 1.60)]))
        noise_std = float(rng.uniform(8.0, 22.0))
        noise = rng.normal(0.0, noise_std, size=region.shape)
        adjusted = (region - 127.5) * contrast + 127.5 + noise
        manipulated[top : top + box_height, left : left + box_width] = np.clip(
            adjusted, 0, 255
        ).astype(np.uint8)
        mask[top : top + box_height, left : left + box_width] = 1
        severity = min(1.0, noise_std / 22.0 * 0.5 + abs(contrast - 1.0))
        metadata.update({"contrast": contrast, "noise_std": noise_std})

    return TamperResult(
        image=manipulated,
        mask=mask,
        tamper_type=tamper_type,
        severity=float(severity),
        metadata=metadata,
    )
