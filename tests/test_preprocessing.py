from pathlib import Path

import numpy as np
from PIL import Image

from mri_integrity_lab.preprocessing import load_standardized_image, standardized_pixel_hash


def test_standardization_preserves_content_inside_square_padding(tmp_path: Path) -> None:
    source = np.full((10, 20), 180, dtype=np.uint8)
    path = tmp_path / "wide.png"
    Image.fromarray(source, mode="L").save(path)

    standardized = load_standardized_image(path, image_size=32)

    assert standardized.size == (32, 32)
    pixels = np.asarray(standardized)
    assert pixels[:7].max() == 0
    assert pixels[8:24].mean() > 150


def test_pixel_hash_is_format_independent_after_standardization(tmp_path: Path) -> None:
    source = np.arange(256, dtype=np.uint8).reshape(16, 16)
    png_path = tmp_path / "sample.png"
    tif_path = tmp_path / "sample.tif"
    Image.fromarray(source, mode="L").save(png_path)
    Image.fromarray(source, mode="L").save(tif_path)

    assert standardized_pixel_hash(png_path, image_size=32) == standardized_pixel_hash(
        tif_path, image_size=32
    )

