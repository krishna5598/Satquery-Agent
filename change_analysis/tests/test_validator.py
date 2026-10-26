"""
Unit tests for bi-temporal validation logic.
"""

from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from change_analysis.preprocessing.validator import (
    validate_bitemporal_pair,
    calculate_bounds_overlap,
    extract_metadata,
)


@pytest.fixture
def sample_pair(tmp_path):
    img_a = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b = np.full((100, 100, 3), 120, dtype=np.uint8)
    p_a = tmp_path / "before.png"
    p_b = tmp_path / "after.png"
    Image.fromarray(img_a).save(p_a)
    Image.fromarray(img_b).save(p_b)
    return p_a, p_b


def test_valid_bitemporal_pair(sample_pair):
    p_a, p_b = sample_pair
    res = validate_bitemporal_pair(p_a, p_b)
    assert res.is_valid is True
    assert len(res.errors) == 0
    assert res.before_meta.width == 100
    assert res.after_meta.height == 100


def test_missing_before_image(tmp_path):
    p_a = tmp_path / "nonexistent_before.png"
    p_b = tmp_path / "after.png"
    Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(p_b)

    res = validate_bitemporal_pair(p_a, p_b)
    assert res.is_valid is False
    assert any("does not exist" in err for err in res.errors)


def test_missing_after_image(tmp_path):
    p_a = tmp_path / "before.png"
    p_b = tmp_path / "nonexistent_after.png"
    Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(p_a)

    res = validate_bitemporal_pair(p_a, p_b)
    assert res.is_valid is False
    assert any("does not exist" in err for err in res.errors)


def test_invalid_corrupted_image(tmp_path):
    p_a = tmp_path / "corrupt_before.png"
    p_b = tmp_path / "valid_after.png"
    p_a.write_bytes(b"not an image file data gibberish")
    Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(p_b)

    res = validate_bitemporal_pair(p_a, p_b)
    assert res.is_valid is False
    assert any("Failed to read image" in err for err in res.errors)


def test_dimension_mismatch_warning(tmp_path):
    p_a = tmp_path / "before.png"
    p_b = tmp_path / "after.png"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(p_a)
    Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)).save(p_b)

    res = validate_bitemporal_pair(p_a, p_b)
    assert res.is_valid is True  # Resampling handles it gracefully
    assert any("dimensions differ" in w for w in res.warnings)


def test_geographic_overlap_calculation():
    # Identical bounds -> 1.0 IoU
    b1 = (0.0, 0.0, 10.0, 10.0)
    b2 = (0.0, 0.0, 10.0, 10.0)
    assert calculate_bounds_overlap(b1, b2) == pytest.approx(1.0)

    # Disjoint bounds -> 0.0 IoU
    b3 = (20.0, 20.0, 30.0, 30.0)
    assert calculate_bounds_overlap(b1, b3) == 0.0

    # 50% partial overlap
    b4 = (5.0, 0.0, 15.0, 10.0)
    overlap = calculate_bounds_overlap(b1, b4)
    assert 0.0 < overlap < 1.0
