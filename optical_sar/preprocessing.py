import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio import Affine
from scipy.ndimage import uniform_filter


def apply_lee_filter(band: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Adaptive Lee speckle filtering for SAR amplitude/power data."""
    img = band.astype(np.float32)
    mean = uniform_filter(img, (window_size, window_size))
    sqr_mean = uniform_filter(img**2, (window_size, window_size))
    variance = np.maximum(sqr_mean - mean**2, 0)
    overall_var = np.var(img) + 1e-8
    weights = variance / (variance + overall_var)
    return mean + weights * (img - mean)


def normalize_optical(img: np.ndarray) -> np.ndarray:
    """Normalizes optical reflectance to [0, 1] range using robust percentiles."""
    norm_bands = []
    for b in img:
        p2, p98 = np.percentile(b, 2), np.percentile(b, 98)
        clipped = np.clip(b, p2, p98)
        norm = (clipped - p2) / (p98 - p2 + 1e-6)
        norm_bands.append(norm)
    return np.stack(norm_bands, axis=0).astype(np.float32)


def normalize_sar(img: np.ndarray) -> np.ndarray:
    """Applies Lee filter and converts SAR backscatter to [0, 1]."""
    filtered = []
    for b in img:
        f = apply_lee_filter(b)
        # Log scale if linear power
        if np.nanmax(f) > 1.0:
            f = np.log1p(np.maximum(f, 0))
        p2, p98 = np.percentile(f, 2), np.percentile(f, 98)
        clipped = np.clip(f, p2, p98)
        norm = (clipped - p2) / (p98 - p2 + 1e-6)
        filtered.append(norm)
    return np.stack(filtered, axis=0).astype(np.float32)


def prepare_tensors(optical_path: str, sar_path: str, target_size=(256, 256)):
    """Reads, aligns, resamples, normalizes, and outputs numpy tensors (C, H, W)."""
    with rasterio.open(optical_path) as opt:
        # 1. Read Optical directly to target size
        opt_data = opt.read(
            out_shape=(opt.count, target_size[0], target_size[1]),
            resampling=Resampling.bilinear,
        )[:3]

        # Calculate the exact geospatial transform of this new 256x256 array
        dst_transform = opt.transform * Affine.scale(
            opt.width / target_size[1], opt.height / target_size[0]
        )
        dst_crs = opt.crs

    with rasterio.open(sar_path) as sar:
        # 2. Dynamically reproject SAR to perfectly match the Optical grid, CRS, and target size
        sar_bands_to_read = min(sar.count, 2)
        sar_data = np.zeros(
            (sar_bands_to_read, target_size[0], target_size[1]), dtype=np.float32
        )

        reproject(
            source=rasterio.band(sar, list(range(1, sar_bands_to_read + 1))),
            destination=sar_data,
            src_transform=sar.transform,
            src_crs=sar.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    opt_tensor = normalize_optical(opt_data)
    sar_tensor = normalize_sar(sar_data)

    return opt_tensor, sar_tensor
