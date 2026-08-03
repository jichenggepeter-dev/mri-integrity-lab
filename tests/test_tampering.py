import numpy as np
from PIL import Image

from mri_integrity_lab.tampering import TamperType, apply_tampering


def test_copy_move_is_deterministic_and_returns_a_mask() -> None:
    pixels = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    image = Image.fromarray(pixels, mode="L")

    first = apply_tampering(image, TamperType.COPY_MOVE, seed=42)
    second = apply_tampering(image, TamperType.COPY_MOVE, seed=42)

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)
    assert first.mask.sum() > 0
    assert np.any(first.image != pixels)


def test_all_tamper_types_change_pixels_and_record_metadata() -> None:
    pixels = np.full((64, 64), 120, dtype=np.uint8)
    image = Image.fromarray(pixels, mode="L")

    for tamper_type in TamperType:
        result = apply_tampering(image, tamper_type, seed=7)
        assert result.tamper_type == tamper_type
        assert result.mask.shape == pixels.shape
        assert result.mask.sum() > 0
        assert np.any(result.image != pixels)
        assert 0.0 < result.severity <= 1.0

