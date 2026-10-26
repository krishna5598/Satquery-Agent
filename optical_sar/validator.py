import rasterio
import numpy as np

def validate_optical_sar_pair(optical_path: str, sar_path: str) -> dict:
    """
    Validates geospatial and dimensional compatibility of an Optical-SAR pair.
    Returns status boolean, metadata dict, and list of warnings/errors.
    """
    issues = []
    meta = {}
    
    try:
        with rasterio.open(optical_path) as opt, rasterio.open(sar_path) as sar:
            meta["optical"] = {
                "crs": str(opt.crs),
                "bounds": list(opt.bounds),
                "shape": (opt.height, opt.width),
                "bands": opt.count,
                "dtypes": opt.dtypes
            }
            meta["sar"] = {
                "crs": str(sar.crs),
                "bounds": list(sar.bounds),
                "shape": (sar.height, sar.width),
                "bands": sar.count,
                "dtypes": sar.dtypes
            }

            # 1. CRS Check
            if opt.crs != sar.crs:
                issues.append(f"CRS mismatch: Optical is {opt.crs}, SAR is {sar.crs}. Reprojection required.")

            # 2. Bounding Box Overlap Check
            opt_b, sar_b = opt.bounds, sar.bounds
            x_overlap = max(0, min(opt_b.right, sar_b.right) - max(opt_b.left, sar_b.left))
            y_overlap = max(0, min(opt_b.top, sar_b.top) - max(opt_b.bottom, sar_b.bottom))
            
            if x_overlap == 0 or y_overlap == 0:
                issues.append("Images do not geographically overlap.")
                return {"valid": False, "metadata": meta, "issues": issues}

            # 3. Shape/Dimension Check
            if (opt.height, opt.width) != (sar.height, sar.width):
                issues.append(f"Dimension mismatch: Optical {opt.height}x{opt.width} vs SAR {sar.height}x{sar.width}. Resampling needed.")

            # 4. Band verification
            if opt.count < 3:
                issues.append(f"Optical expected >= 3 bands (RGB), found {opt.count}.")
            if sar.count < 1:
                issues.append("SAR raster contains 0 bands.")

    except Exception as e:
        return {"valid": False, "metadata": {}, "issues": [f"File read error: {str(e)}"]}

    is_valid = len(issues) == 0 or all("Resampling" in s or "Reprojection" in s for s in issues)
    return {
        "valid": is_valid,
        "metadata": meta,
        "warnings": issues
    }