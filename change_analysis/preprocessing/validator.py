"""
Bi-temporal Input Validator for SatQuery AI Change Analysis.

Validates image existence, readable formats, dimensions, channels, datatypes,
NaN/Inf anomalies, and geospatial correspondence (CRS, bounds, overlap, resolution).
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

try:
    import tifffile
except ImportError:
    tifffile = None


@dataclass
class ImageMetadata:
    path: str
    format: str
    width: int
    height: int
    channels: int
    dtype: str
    has_geospatial: bool = False
    crs: Optional[str] = None
    bounds: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy)
    resolution: Optional[Tuple[float, float]] = None            # (res_x, res_y)
    acquisition_date: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    before_meta: Optional[ImageMetadata] = None
    after_meta: Optional[ImageMetadata] = None
    geographic_overlap: Optional[float] = None  # Jaccard / Overlap fraction (0.0 to 1.0)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def extract_metadata(image_path: str | Path) -> Tuple[Optional[ImageMetadata], List[str], Optional[np.ndarray]]:
    """Inspects an image file and extracts dimensional, statistical, and geospatial metadata."""
    path = Path(image_path)
    errors = []

    if not path.exists():
        return None, [f"Image file does not exist: {path}"], None

    if not path.is_file():
        return None, [f"Path is not a regular file: {path}"], None

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None, [f"Unsupported file format '{ext}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"], None

    # Load image array safely
    array: Optional[np.ndarray] = None
    format_name = ext.lstrip(".").upper()
    tags: Dict[str, Any] = {}
    crs: Optional[str] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    resolution: Optional[Tuple[float, float]] = None
    has_geo = False
    acq_date: Optional[str] = None

    try:
        if ext in {".tif", ".tiff"} and tifffile is not None:
            with tifffile.TiffFile(str(path)) as tif:
                array = tif.asarray()
                # Inspect TIFF tags for GeoTIFF metadata
                first_page = tif.pages[0]
                for tag in first_page.tags:
                    tags[tag.name] = str(tag.value)[:100]

                # Extract GeoTIFF geokeys if present
                if "ModelPixelScaleTag" in first_page.tags:
                    has_geo = True
                    scale = first_page.tags["ModelPixelScaleTag"].value
                    resolution = (float(scale[0]), float(scale[1]))

                if "ModelTiepointTag" in first_page.tags and resolution is not None:
                    tp = first_page.tags["ModelTiepointTag"].value
                    # [i, j, k, x, y, z] -> tie point mapping
                    if len(tp) >= 6:
                        origin_x, origin_y = tp[3], tp[4]
                        h, w = array.shape[:2]
                        bounds = (
                            float(origin_x),
                            float(origin_y - h * resolution[1]),
                            float(origin_x + w * resolution[0]),
                            float(origin_y)
                        )

                if "GeoKeyDirectoryTag" in first_page.tags:
                    crs = "GeoKey Directory Tag Present (TIFF CRS)"
                    
                if "DateTime" in first_page.tags:
                    acq_date = str(first_page.tags["DateTime"].value)
        else:
            with Image.open(path) as img:
                format_name = img.format or format_name
                # Check EXIF acquisition date if present
                exif = img.getexif()
                if exif and 306 in exif:  # 306 is DateTime in EXIF
                    acq_date = str(exif[306])
                array = np.array(img)

    except Exception as e:
        return None, [f"Failed to read image at '{path}': {str(e)}"], None

    if array is None or array.size == 0:
        return None, [f"Image file is empty or corrupted: {path}"], None

    # Determine dimensions & channels
    if array.ndim == 2:
        h, w = array.shape
        channels = 1
    elif array.ndim == 3:
        h, w, channels = array.shape
    else:
        return None, [f"Unsupported array dimension ({array.ndim}D) in {path}"], None

    # Check for NaN / Inf
    if not np.issubdtype(array.dtype, np.integer):
        if np.isnan(array).any():
            errors.append(f"Image contains NaN values: {path}")
        if np.isinf(array).any():
            errors.append(f"Image contains Infinite values: {path}")

    meta = ImageMetadata(
        path=str(path.resolve()),
        format=format_name,
        width=w,
        height=h,
        channels=channels,
        dtype=str(array.dtype),
        has_geospatial=has_geo,
        crs=crs,
        bounds=bounds,
        resolution=resolution,
        acquisition_date=acq_date,
        tags=tags,
    )
    return meta, errors, array


def calculate_bounds_overlap(
    bounds1: Tuple[float, float, float, float],
    bounds2: Tuple[float, float, float, float]
) -> float:
    """Calculates the Intersection-over-Union overlap fraction between two geographic bounding boxes."""
    minx1, miny1, maxx1, maxy1 = bounds1
    minx2, miny2, maxx2, maxy2 = bounds2

    inter_minx = max(minx1, minx2)
    inter_miny = max(miny1, miny2)
    inter_maxx = min(maxx1, maxx2)
    inter_maxy = min(maxy1, maxy2)

    if inter_maxx <= inter_minx or inter_maxy <= inter_miny:
        return 0.0

    inter_area = (inter_maxx - inter_minx) * (inter_maxy - inter_miny)
    area1 = (maxx1 - minx1) * (maxy1 - miny1)
    area2 = (maxx2 - minx2) * (maxy2 - miny2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def validate_bitemporal_pair(
    before_path: str | Path,
    after_path: str | Path,
    verify_crs_bounds: bool = True
) -> ValidationResult:
    """
    Validates a bi-temporal satellite image pair (Before & After).

    Detects:
    - Missing or corrupted files
    - Dimension mismatches
    - Channel mismatches
    - Geospatial bounds mismatch / disjoint geography
    - Acquisition date anomalies
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Inspect Before Image
    before_meta, before_errs, before_arr = extract_metadata(before_path)
    errors.extend(before_errs)

    # 2. Inspect After Image
    after_meta, after_errs, after_arr = extract_metadata(after_path)
    errors.extend(after_errs)

    if before_meta is None or after_meta is None:
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # 3. Check Dimensions
    if (before_meta.width, before_meta.height) != (after_meta.width, after_meta.height):
        warnings.append(
            f"Image dimensions differ: BEFORE is ({before_meta.width}x{before_meta.height}) "
            f"while AFTER is ({after_meta.width}x{after_meta.height}). Automatic alignment/rescaling will occur."
        )

    # 4. Check Channels
    if before_meta.channels != after_meta.channels:
        warnings.append(
            f"Channel counts differ: BEFORE has {before_meta.channels} channels while AFTER has {after_meta.channels} channels. "
            "Normalization will conform both to standard 3-channel RGB."
        )

    # 5. Check Temporal Metadata (Acquisition dates)
    if before_meta.acquisition_date and after_meta.acquisition_date:
        if before_meta.acquisition_date > after_meta.acquisition_date:
            warnings.append(
                f"Potential temporal inversion: BEFORE date ({before_meta.acquisition_date}) is later than "
                f"AFTER date ({after_meta.acquisition_date}). Verify image ordering."
            )

    # 6. Check Geospatial Correspondence
    overlap_fraction: Optional[float] = None
    if verify_crs_bounds and before_meta.has_geospatial and after_meta.has_geospatial:
        if before_meta.crs != after_meta.crs and (before_meta.crs and after_meta.crs):
            warnings.append(
                f"CRS mismatch: BEFORE CRS is '{before_meta.crs}' but AFTER CRS is '{after_meta.crs}'. "
                "Reprojection may be required for exact geospatial alignment."
            )

        if before_meta.bounds and after_meta.bounds:
            overlap = calculate_bounds_overlap(before_meta.bounds, after_meta.bounds)
            overlap_fraction = round(overlap, 4)
            if overlap == 0.0:
                errors.append(
                    f"Geographic mismatch: BEFORE bounds {before_meta.bounds} and AFTER bounds {after_meta.bounds} "
                    "have 0% spatial overlap! They appear to represent entirely different geographic locations."
                )
            elif overlap < 0.70:
                warnings.append(
                    f"Low spatial overlap ({overlap_fraction * 100:.1f}%). Ensure regions correspond to the same target area."
                )

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        before_meta=before_meta,
        after_meta=after_meta,
        geographic_overlap=overlap_fraction,
    )
