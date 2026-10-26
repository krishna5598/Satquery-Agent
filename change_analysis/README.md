# SatQuery AI — Change Analysis Specialist Module (Member 3)

Autonomous, reproducible, and demo-ready multimodal remote-sensing change analysis system designed for the **SatQuery AI** agentic multimodal framework.

The Change Analysis module operates as the visual evidence and change comprehension specialist in the overall architecture:
```
USER QUERY
    ↓
VALIDATION & ROUTING
    ↓
CHANGE ANALYSIS SPECIALIST (change_tool)
    ↓
ChangeCLIP / ChangeChat / Fallback Engine
    ↓
STRUCTURED EVIDENCE (Change Maps, Overlays, Metrics)
    ↓
LLM EXPLANATION (Grounded Reasoning)
    ↓
FINAL ANSWER
```

---

## 1. Features & Capabilities

- **Bi-Temporal Validation**: Verifies input image existence, readable formats, dimensional consistency, channel counts, data types, NaN/Inf values, and geospatial bounding-box overlap.
- **Radiometric & Spatial Preprocessing**: Automatically handles multi-band scaling, spatial resampling, optional histogram matching, and reproducible PyTorch tensor generation.
- **ChangeCLIP Runner**: Implements the dual-temporal 6-channel CLIP difference architecture for pixel-level change detection.
- **ChangeChat Runner**: Implements multimodal instruction-tuned conversational VQA for answering both general ("What changed?") and category-specific queries ("Has construction increased?").
- **Deterministic Fallback Engine (`deterministic_fallback`)**: A mathematically rigorous CPU-executable engine combining Structural Similarity Index (SSIM), multi-channel spectral difference, visible vegetation ($\Delta\text{NDVI}$ proxy) & water ($\Delta\text{NDWI}$ proxy) shifts, Otsu adaptive thresholding, morphological filtering, and connected-component cluster extraction.
- **Standardized `change_tool()` Interface**: Implements the shared machine-readable JSON schema for seamless LangGraph node integration.
- **Publication-Quality Visual Artifacts**: Generates binary change masks, continuous probability heatmaps, transparent post-change overlays, and comprehensive 4-panel comparison figures.
- **Automated Benchmark Evaluation**: Evaluates predicted masks against ground truth masks with IoU (Jaccard), F1-Score (Dice), Precision, Recall, Overall Accuracy, and Cohen's Kappa.
- **Local FastAPI Microservice**: Lightweight REST API with `/health` and `/analyze-change` endpoints.

---

## 2. Architecture & Design Principles

```
change_analysis/
│
├── README.md                          # Comprehensive documentation (this file)
├── MODEL_INFO.md                      # Verified scientific facts on ChangeCLIP & ChangeChat
├── requirements.txt                   # Production dependencies
├── requirements-gpu.txt               # GPU-specific notes
├── .gitignore                         # Git hygiene rules
├── config.yaml                        # Configuration for models, thresholds, and paths
├── config.py                          # Pydantic/dataclass configuration loader
├── hardware_check.py                  # Standalone hardware detection utility
├── cli.py                             # Full CLI interface
├── api.py                             # FastAPI local REST service
├── demo.py                            # Standalone demo running all 4 cases
│
├── preprocessing/
│   ├── __init__.py
│   ├── validator.py                   # Bi-temporal validation & geospatial bounds check
│   ├── alignment.py                   # Spatial co-registration & dimension resampling
│   ├── normalization.py               # Radiometric balancing & dynamic range scaling
│   └── transforms.py                  # PyTorch model tensor preparation
│
├── models/
│   ├── __init__.py
│   ├── changeclip_runner.py           # ChangeCLIP inference runner
│   ├── changechat_runner.py           # ChangeChat multimodal conversational runner
│   └── fallback_runner.py             # Deterministic spectral & morphological engine
│
├── pipeline/
│   ├── __init__.py
│   └── change_pipeline.py             # Unified pipeline orchestration
│
├── tools/
│   ├── __init__.py
│   └── change_tool.py                 # SatQuery AI specialist tool interface
│
├── integrations/
│   ├── __init__.py
│   └── langgraph_change_tool.py       # LangGraph StateGraph node adapter
│
├── visualization/
│   ├── __init__.py
│   └── change_map.py                  # Heatmaps, binary masks, and comparison quads
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                     # IoU, F1, Precision, Recall, Accuracy, Kappa
│   └── evaluate.py                    # Evaluation runner with ground truth comparison
│
├── examples/
│   ├── README.md                      # Dataset provenance and licensing
│   ├── case_01/                       # Construction / built-up expansion (LEVIR-CD)
│   ├── case_02/                       # Water change (Inundation / reservoir rise)
│   ├── case_03/                       # Vegetation clearing / deforestation
│   └── case_no_change/                # Scene stability baseline (0% change)
│
├── outputs/                           # Saved visual figures and JSON result logs
│
└── tests/                             # Automated test suite (19 unit/integration tests)
```

---

## 3. Hardware Requirements & Detection

The module automatically detects host capabilities and adapts its execution mode:

| Component | Minimum (Fallback Mode) | Recommended (Real GPU Mode) |
| :--- | :--- | :--- |
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS | Linux x86_64 / Windows |
| **CPU** | Dual-core 2.0GHz+ | Quad-core 3.0GHz+ |
| **RAM** | $\ge$4 GB | $\ge$16 GB |
| **GPU** | Integrated (CPU Execution) | NVIDIA GPU with $\ge$8 GB VRAM (ChangeCLIP) / $\ge$48 GB (ChangeChat) |
| **CUDA** | None | CUDA 11.7+ |

### Run Hardware Inspection
```powershell
# Windows (PowerShell)
python change_analysis/hardware_check.py

# Linux / macOS
python3 change_analysis/hardware_check.py
```

Sample output:
```
============================================================
 SatQuery AI - Hardware & Environment Inspection Report
============================================================
 Operating System  : Windows 11 (10.0.26100)
 Python Version    : 3.13.14
 PyTorch Version   : 2.6.0+cpu
 CPU               : Intel64 Family 6 Model 142 Stepping 9 (4 threads)
 System RAM        : 7.87 GB total (1.54 GB available)
 CUDA Available    : NO
 GPU Name          : Intel(R) HD Graphics 620
 GPU VRAM          : 0.0 GB
 Selected Device   : cpu
 Recommended Mode  : FALLBACK / CPU
------------------------------------------------------------
 Assessment        : Host machine has 7.87GB RAM and no CUDA acceleration.
                     Deterministic fallback is strongly recommended for responsive demo & evaluation.
============================================================
```

---

## 4. Installation & Setup

### 1. Clone & Enter Workspace
```powershell
cd "d:\Akhi\Projects\SatQuery AI"
```

### 2. Install Dependencies
```powershell
# Core CPU dependencies (Standard)
pip install -r change_analysis/requirements.txt

# Optional: GPU acceleration (if NVIDIA CUDA is available)
pip install -r change_analysis/requirements-gpu.txt
```

---

## 5. Model Setup & Provenance

Detailed scientific facts regarding **ChangeCLIP** and **ChangeChat** are recorded in [`change_analysis/MODEL_INFO.md`](file:///d:/Akhi/Projects/SatQuery%20AI/change_analysis/MODEL_INFO.md).

- **ChangeCLIP**: Official repository `dyzy41/ChangeCLIP`. Uses dual 6-channel CLIP visual backbone and context decoder. Fine-tuned checkpoints are hosted via Baidu Netdisk by the authors. To use custom weights, place them in `change_analysis/checkpoints/changeclip_levir.pth` and set `changeclip.checkpoint_path` in `config.yaml`.
- **ChangeChat**: Official repository `hanlinwu/ChangeChat`. Architecture is a 7B parameter multimodal LLaVA VLM requiring $\ge$48GB VRAM. Pre-trained weights are marked *"coming soon"* by the authors.
- **Fallback Engine**: Automatically active when running on consumer hardware without CUDA or when checkpoints are absent. Fully functional, transparently labeled `"model": "deterministic_fallback"`.

---

## 6. How to Run

### Run All 4 Benchmark Demo Cases
Runs all 4 cases autonomously, outputs visual comparison quads to `outputs/`, calculates metrics, and prints an executive summary:
```powershell
# Windows PowerShell
python demo.py

# Or via module path
python change_analysis/demo.py
```

### Run CLI for a Single Image Pair
```powershell
# Run change analysis on any bitemporal pair with custom query
python -m change_analysis.cli `
  --before change_analysis/examples/case_01/before.png `
  --after change_analysis/examples/case_01/after.png `
  --question "Has construction increased?"

# Run with ground-truth evaluation
python -m change_analysis.cli `
  --before change_analysis/examples/case_01/before.png `
  --after change_analysis/examples/case_01/after.png `
  --question "What changed?" `
  --ground-truth change_analysis/examples/case_01/ground_truth.png
```

### Run Local REST API Service
```powershell
python change_analysis/api.py
```
- Health Check: `GET http://127.0.0.1:8000/health`
- Inference by Path: `POST http://127.0.0.1:8000/analyze-change-by-path`
- File Upload: `POST http://127.0.0.1:8000/analyze-change` (multipart/form-data)

---

## 7. Specialist Tool Contract (`change_tool`)

The `change_tool()` function provides the standardized interface for SatQuery AI:

```python
from change_analysis.tools.change_tool import change_tool

result = change_tool(
    before_image="change_analysis/examples/case_01/before.png",
    after_image="change_analysis/examples/case_01/after.png",
    question="Has construction increased?"
)
```

### Output Schema
```json
{
  "task": "bi_temporal_change",
  "model": "deterministic_fallback",
  "answer": "Yes, visible built-up expansion is confirmed. Significant structural and albedo shifts were detected across 17.95% of the scene (55 distinct clusters). New structures exhibit increased surface reflectance (mean brightness delta: +0.287).",
  "confidence": null,
  "evidence": [
    {
      "type": "change_map",
      "path": "D:/Akhi/Projects/SatQuery AI/change_analysis/outputs/case_01_change_map.png",
      "description": "Binary raster mask indicating detected pixel-level change footprints."
    },
    {
      "type": "overlay",
      "path": "D:/Akhi/Projects/SatQuery AI/change_analysis/outputs/case_01_overlay.png",
      "description": "Composite visualization with changes highlighted in red over the post-change acquisition."
    },
    {
      "type": "comparison",
      "path": "D:/Akhi/Projects/SatQuery AI/change_analysis/outputs/case_01_comparison.png",
      "description": "4-panel demonstration figure: Before (T1), After (T2), Change Mask, and Change Overlay."
    }
  ],
  "metrics": {
    "changed_area_fraction": 0.1795,
    "changed_pixel_count": 11765,
    "total_pixel_count": 65536,
    "changed_regions_count": 55,
    "dominant_change_type": "built_up",
    "mean_delta_vegetation_index": -0.0421,
    "mean_delta_water_index": -0.0152,
    "mean_delta_brightness": 0.2871,
    "mean_ssim_difference": 0.4561
  },
  "warnings": [],
  "parameters": {
    "before": "...",
    "after": "...",
    "question": "Has construction increased?",
    "device": "cpu",
    "elapsed_seconds": 0.84
  }
}
```

---

## 8. Demonstration Benchmark Results

| Case | Scenario | Changed % | IoU (Jaccard) | F1 (Dice) | Accuracy | Model Used |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Case 01** | Construction / Built-up Expansion (LEVIR-CD) | 17.95% | 0.1368 | 0.2407 | 67.25% | `deterministic_fallback` |
| **Case 02** | Water Inundation / Reservoir Rise | 18.26% | 0.9964 | 0.9982 | 99.93% | `deterministic_fallback` |
| **Case 03** | Vegetation Clearing / Canopy Loss | 17.54% | 0.9994 | 0.9997 | 99.99% | `deterministic_fallback` |
| **Case 04** | No Significant Change (Scene Stability) | 0.00% | 1.0000 | 0.0000 | 100.00% | `deterministic_fallback` |

---

## 9. Automated Testing

Run the full pytest test suite (19 automated unit & integration tests):
```powershell
python -m pytest change_analysis/tests -v
```

Test coverage includes:
- Input validation (missing files, invalid formats, dimension mismatches, geographic bounds overlap)
- Preprocessing (radiometric normalization, dynamic range stretch, spatial alignment, model tensors)
- Deterministic fallback engine (identical scenes, high-contrast changes, vegetation index shift)
- Contract compliance (schema validation, null confidence verification, evidence paths existence)
- Ground-truth evaluation metrics (IoU, F1, Accuracy, Precision, Recall edge cases)
- End-to-end pipeline execution on real bitemporal samples

---

## 10. LangGraph Integration Guide

The module includes a pre-built node adapter in [`change_analysis/integrations/langgraph_change_tool.py`](file:///d:/Akhi/Projects/SatQuery%20AI/change_analysis/integrations/langgraph_change_tool.py).

To plug into SatQuery AI's central LangGraph workflow:
```python
from langgraph.graph import StateGraph
from change_analysis.integrations.langgraph_change_tool import make_langgraph_change_node

# In your master SatQuery AI workflow:
workflow = StateGraph(...)
workflow.add_node("change_analysis", make_langgraph_change_node())
# Wire router edge to 'change_analysis' when query targets temporal change
```

---

## 11. Licenses & Citations

- **Code**: MIT License.
- **ChangeCLIP**: Apache 2.0 (OpenMMLab MMSegmentation).
- **ChangeChat**: Research License (LLaVA / Vicuna).
- **LEVIR-CD Dataset**: Open Access Research License (Chen & Shi, 2020).
