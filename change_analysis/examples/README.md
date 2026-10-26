# Demonstration Datasets & Provenance

This directory contains four reliable, reproducible bitemporal satellite image demonstration cases for the SatQuery AI Change Analysis module.

---

## Overview of Cases

| Case | Scenario | Primary Signature | Ground Truth Mask | Source / Benchmark | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Case 01** | Construction / Built-up Expansion | New residential and commercial buildings erected; high-reflectance roof structures | `ground_truth.png` (25.18% change) | LEVIR-CD Benchmark (`test_2_0000_0000.png`) | Research / Open Access |
| **Case 02** | Water Change | River channel and reservoir inundation / water level rise ($\Delta\text{NDWI} > +0.15$) | `ground_truth.png` (18.30% change) | Multi-temporal remote sensing water dataset | Open Access |
| **Case 03** | Vegetation Clearing | Forest clearing and canopy removal ($\Delta\text{NDVI} < -0.15$) | `ground_truth.png` (17.50% change) | Change Detection Benchmark (`test_77_0512_0256.png`) | Research / Open Access |
| **Case 04** | No Significant Change | Stable landscape with sensor radiometric variance and illumination shifts | `ground_truth.png` (0.00% change) | Bitemporal stable baseline (`train_386_0512_0768.png`) | Research / Open Access |

---

## File Structure per Case
Each case folder contains:
- `before.png`: Pre-change temporal acquisition ($T_1$, $256 \times 256 \times 3$ RGB)
- `after.png`: Post-change temporal acquisition ($T_2$, $256 \times 256 \times 3$ RGB)
- `ground_truth.png`: Binary annotation mask ($0$ = background, $255$ = true changed pixels)

---

## Data Licenses & Citation
1. **LEVIR-CD**:
   - Hao Chen and Zhenwei Shi. *"A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection."* Remote Sensing, 2020.
2. **BIT_CD / STANet Samples**:
   - Justchenhao remote sensing repository ([https://github.com/justchenhao/BIT_CD](https://github.com/justchenhao/BIT_CD)).
