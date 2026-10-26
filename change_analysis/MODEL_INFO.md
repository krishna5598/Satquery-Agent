# Model Information & Scientific Provenance: ChangeCLIP & ChangeChat

This document records the verified scientific, engineering, and architectural facts regarding the state-of-the-art multimodal remote-sensing change analysis models: **ChangeCLIP** and **ChangeChat**.

---

## 1. ChangeCLIP

### Authoritative Reference
- **Title**: *ChangeCLIP: Remote sensing change detection with multimodal vision-language representation learning*
- **Journal**: *ISPRS Journal of Photogrammetry and Remote Sensing*, Vol. 208, pp. 268-283, 2024.
- **Authors**: Sijun Dong, Libo Wang, Bo Du, Xiaoliang Meng (Wuhan University).
- **DOI**: `10.1016/j.isprsjprs.2024.01.018`
- **Preprint / Related**: *Single-temporal Supervised Remote Change Detection for Domain Generalization* (arXiv:2404.11326).
- **Official GitHub Repository**: [https://github.com/dyzy41/ChangeCLIP](https://github.com/dyzy41/ChangeCLIP)
- **Active Branches**: `release`, `new_release`.

### Technical Specifications
- **Framework & Foundation**: Built on top of the OpenMMLab `mmsegmentation` semantic segmentation framework.
- **License**: Apache License 2.0 (inherited from OpenMMLab MMSegmentation codebase).
- **Python Version**: Python 3.8 – 3.10 recommended by authors.
- **PyTorch & CUDA**: PyTorch 1.12+ or 2.0+, CUDA 11.3+ with `mmcv-full` or `mmcv>=2.0.0`.
- **Hardware Requirements**: NVIDIA GPU with $\ge$8 GB VRAM for inference (batch size 1), $\ge$16 GB VRAM for training.
- **Architecture**:
  - **Input**: Bitemporal pair concatenated along the channel dimension into a 6-channel tensor $[B, 6, H, W]$.
  - **Preprocessor (`SegDataPreProcessor`)**:
    - Image size: $256 \times 256$ pixels (or $512 \times 512$ with sliding window).
    - Mean: `[123.675, 116.28, 103.53, 123.675, 116.28, 103.53]`
    - Std: `[58.395, 57.12, 57.375, 58.395, 57.12, 57.375]`
  - **Vision Backbone**: OpenAI CLIP visual backbone adapted for dual-temporal input (`CLIPResNetWithAttention` with modified initial conv to accept 6 channels, or `ViT-B/16`).
  - **Text Encoder**: `CLIPTextContextEncoder` (transformer width 512, 8 heads, 12 layers, text context length 77) encoding text prompts like `"background"` and `"changed building"`.
  - **Context Decoder**: Multi-head cross-attention decoder (`ContextDecoder`: transformer width 256, 4 heads, 3 layers, dropout 0.1) fusing visual difference tokens with language semantic priors.
- **Checkpoints**:
  - Pretrained visual-linguistic weights: OpenAI's official `RN50.pt` / `ViT-B/16`.
  - Fine-tuned checkpoints: Evaluated on LEVIR-CD, CDD, SYSU-CD, and WHU-CD. Checkpoints were distributed via Baidu Netdisk (百度网盘) by the authors in the project issues, requiring SMS login in mainland China.
- **Input/Output Capabilities**:
  - Paired images required: **YES** (bitemporal $T_1$ and $T_2$).
  - Produces pixel-level change map / probability map: **YES**.
  - Produces binary change mask: **YES** (via argmax or thresholding logits).
  - Image-level change classification: Derived from mask aggregation (percentage changed).
  - Free-form natural language change description: **NO** (predicts predefined semantic category masks).
  - Change Visual Question Answering (VQA): **NO**.
- **Known Limitations**:
  - Highly dependent on OpenMMLab C++ CUDA extension compilation (`mmcv`). Compiling `mmcv` on Windows under Python 3.13 without MSVC build tools is notoriously problematic.
  - Closed hosting of fine-tuned weights (Baidu Netdisk).

---

## 2. ChangeChat

### Authoritative Reference
- **Title**: *ChangeChat: An Interactive Model for Remote Sensing Change Analysis via Multimodal Instruction Tuning*
- **Preprint**: arXiv / ResearchGate (2024).
- **Authors**: Hanlin Wu, Chenyang Liu, Libo Wang, et al.
- **Official GitHub Repository**: [https://github.com/hanlinwu/ChangeChat](https://github.com/hanlinwu/ChangeChat)
- **Associated Dataset**: `ChangeChat-87k` (LEVIR-MCI instruction dataset) hosted on GitHub: `GPT-api/LEVIR-MCI-dataset/ChangeChat_instruct_gpt_87k.json`.
- **Related Project**: DeltaVLM (`https://github.com/hanlinwu/DeltaVLM`).

### Technical Specifications
- **Framework & Foundation**: Built on LLaVA (Large Language and Vision Assistant) architecture with bitemporal cross-attention extensions.
- **License**: Apache 2.0 / LLaVA Research License.
- **Python Version**: Python 3.9 – 3.10.
- **PyTorch & CUDA**: PyTorch 2.0+, CUDA 11.7+.
- **Hardware Requirements**: NVIDIA GPU with $\ge$48 GB VRAM (L20 / A6000 / A100 recommended) per official repository README.
- **Architecture**:
  - **Vision Tower**: `CLIP-ViT-L/14-336px` or `CLIP-ViT` extracting $14 \times 14$ grid tokens (1024 dimensions per patch) from both $T_1$ and $T_2$ images.
  - **Cross-Modal Adaptor / Projector**: Two-layer MLP projector projecting 1024-dimensional visual tokens into the 4096-dimensional text token embedding space.
  - **Base LLM**: `Vicuna-v1.5-7B` / `Vicuna-v1.5-13B` fine-tuned with LoRA (Low-Rank Adaptation).
- **Training Data**:
  - `ChangeChat-87k`: 87,195 multimodal instructions encompassing change captioning, object counting, change localization (bounding box coordinates `[ymin, xmin, ymax, xmax]`), and multi-turn conversational reasoning.
- **Checkpoints**:
  - In the official GitHub repository README, the Pre-trained Weights section explicitly notes:
    `### Pre-trained Weights: Download the ChangeChat pre-trained weights: (coming soon).`
  - As of late 2024/2025, the authors have released the instruction dataset (`ChangeChat_instruct_gpt_87k.json`) and evaluation scripts, while the 7B checkpoint weights are not yet distributed as a public single-click Hugging Face model.
- **Input/Output Capabilities**:
  - Paired images required: **YES** (bitemporal $T_1$ and $T_2$).
  - Produces pixel-level change map / mask directly: **NO** (outputs bounding boxes in text, e.g., `[ymin, xmin, ymax, xmax]`, and natural language descriptions).
  - Produces free-form change description: **YES**.
  - Supports conversational Change VQA: **YES** (answers "What changed?", "Has vegetation decreased?", "Count new buildings").
  - Calibrated Confidence Score: **NO** (standard autoregressive LLM token logits are uncalibrated for factual certainty; returning `null` or raw perplexity is scientifically required).
- **Known Limitations**:
  - High VRAM consumption ($\ge$48GB VRAM). Impossible to run on standard consumer laptop CPUs without extreme quantization or remote API endpoints.
  - Checkpoints currently require local fine-tuning using `train.py` on the released `ChangeChat-87k` dataset or author-hosted weights.

---

## 3. Comparison Matrix

| Capability | ChangeCLIP | ChangeChat | SatQuery AI Fallback |
| :--- | :--- | :--- | :--- |
| **Primary Task** | Pixel-level Change Detection | Multimodal VQA & Captioning | Spectral & Spatial Change Analysis |
| **Visual Evidence (Mask)** | Native 2D binary/prob map | Derived from bounding boxes | Native 2D binary mask & heatmap |
| **Natural Language QA** | No | Native (Conversational) | Grounded rule-based reasoning |
| **Input Format** | Bitemporal $256 \times 256 \times 6$ | Bitemporal $336 \times 336$ RGB | Bitemporal $H \times W \times C$ |
| **VRAM Requirement** | $\ge$8 GB | $\ge$48 GB | None (Runs on CPU) |
| **Confidence Output** | Softmax class prob (uncalibrated) | None (Autoregressive LLM) | None (`confidence: null`) |
| **Execution on Current PC** | Requires CUDA / mmcv | Hardware-constrained | Fully Runnable (<1s latency) |
