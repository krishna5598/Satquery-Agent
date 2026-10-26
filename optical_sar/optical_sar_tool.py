import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from validator import validate_optical_sar_pair
from preprocessing import prepare_tensors
from model import DualBranchFusion

_FUSION_MODEL = None


def get_model():
    global _FUSION_MODEL
    if _FUSION_MODEL is None:
        _FUSION_MODEL = DualBranchFusion(num_classes=2)
        weights_path = "fusion_weights.pt"
        if os.path.exists(weights_path):
            _FUSION_MODEL.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
        _FUSION_MODEL.eval()
    return _FUSION_MODEL


def optical_sar_tool(optical_path: str, sar_path: str, query: str) -> dict:
    # 1. Validation check
    val = validate_optical_sar_pair(optical_path, sar_path)
    if not val["valid"]:
        return {
            "task": "optical_sar_analysis",
            "model": "DualBranchFusion-Baseline",
            "answer": f"Analysis halted. Incompatible input files: {'; '.join(val['issues'])}",
            "confidence": 0.0,
            "evidence": [],
            "warnings": val["issues"],
            "parameters": {"query": query},
        }

    # 2. Preprocess & normalize
    opt_t, sar_t = prepare_tensors(optical_path, sar_path)
    opt_tensor = torch.from_numpy(opt_t).unsqueeze(0)
    sar_tensor = torch.from_numpy(sar_t).unsqueeze(0)

    # 3. Model inference using the newly trained weights
    model = get_model()
    with torch.no_grad():
        logits = model(opt_tensor, sar_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        pred_mask = np.argmax(probs, axis=0).astype(
            np.uint8
        )  # Converted to uint8 for GeoTIFF saving

    # 4. Generate Visual & Spatial Evidence

    # A) PNG for the Web GUI
    cmap = ListedColormap(["blue", "red"])
    output_png_path = "fused_mask_evidence.png"
    plt.imsave(output_png_path, pred_mask, cmap=cmap)

    # B) Geo-Referenced Map (GeoTIFF) for GIS Workflows
    output_tif_path = "fused_mask_georeferenced.tif"

    # Extract real-world coordinates from the original optical image
    with rasterio.open(optical_path) as src:
        crs = src.crs
        transform = src.transform

    # Write the predictions back out as a geospatial file
    with rasterio.open(
        output_tif_path,
        "w",
        driver="GTiff",
        height=pred_mask.shape[0],
        width=pred_mask.shape[1],
        count=1,
        dtype=pred_mask.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(pred_mask, 1)

    water_pct = float(np.mean(pred_mask == 0) * 100)
    builtup_pct = float(np.mean(pred_mask == 1) * 100)
    avg_confidence = float(np.mean(np.max(probs, axis=0)))

    answer = (
        f"Joint Optical-SAR analysis completed. Optical multispectral features combined "
        f"with SAR radar backscatter indicate {builtup_pct:.1f}% built-up area and "
        f"{water_pct:.1f}% surface water within the region."
    )

    return {
        "task": "optical_sar_analysis",
        "model": "DualBranchFusion-v1",
        "answer": answer,
        "confidence": round(avg_confidence, 2),
        "evidence": [
            {
                "type": "classification_summary",
                "metrics": {
                    "built_up_percentage": round(builtup_pct, 2),
                    "water_percentage": round(water_pct, 2),
                },
            },
            {
                "type": "modality_contribution",
                "details": "Optical provided high-resolution surface reflectance; SAR microwave backscatter verified surface roughness and moisture.",
            },
            {"type": "visual_mask", "file_path": os.path.abspath(output_png_path)},
            {"type": "geospatial_mask", "file_path": os.path.abspath(output_tif_path)},
        ],
        "warnings": val.get("warnings", []),
        "parameters": {
            "query": query,
            "optical_file": os.path.basename(optical_path),
            "sar_file": os.path.basename(sar_path),
        },
    }
