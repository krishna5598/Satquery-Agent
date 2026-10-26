"""
Configuration loader and validator for SatQuery AI Change Analysis module.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Dict
import yaml

from change_analysis.hardware_check import detect_hardware


@dataclass
class SystemConfig:
    mode: str = "auto"
    device: str = "auto"
    seed: int = 42
    log_level: str = "INFO"


@dataclass
class PathsConfig:
    output_dir: str = "outputs"
    checkpoints_dir: str = "checkpoints"
    examples_dir: str = "examples"


@dataclass
class PreprocessingConfig:
    target_size: List[int] = field(default_factory=lambda: [256, 256])
    enforce_square: bool = False
    max_dimension: int = 1024
    check_nan_inf: bool = True
    verify_crs_bounds: bool = True
    clip_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    clip_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ChangeCLIPConfig:
    model_name: str = "ChangeCLIP"
    backbone: str = "RN50"
    checkpoint_path: str = "checkpoints/changeclip_levir.pth"
    threshold: float = 0.5
    allow_cpu_inference: bool = True


@dataclass
class ChangeChatConfig:
    model_name: str = "ChangeChat"
    model_base: str = "lmsys/vicuna-7b-v1.5"
    checkpoint_path: str = "checkpoints/changechat_lora"
    max_new_tokens: int = 256
    temperature: float = 0.2
    allow_cpu_inference: bool = False


@dataclass
class FallbackConfig:
    algorithm: str = "deterministic_spectral_structural"
    ssim_weight: float = 0.35
    spectral_weight: float = 0.65
    otsu_thresholding: bool = True
    fixed_threshold: float = 0.25
    morph_kernel_size: int = 3
    min_cluster_pixels: int = 30
    calculate_indices: bool = True


@dataclass
class AppConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    changeclip: ChangeCLIPConfig = field(default_factory=ChangeCLIPConfig)
    changechat: ChangeChatConfig = field(default_factory=ChangeChatConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    effective_device: str = "cpu"
    effective_mode: str = "fallback"

    def resolve(self) -> AppConfig:
        """Resolves automatic device and mode selection based on hardware inspection."""
        hw = detect_hardware()
        
        # Device resolution
        if self.system.device == "auto":
            self.effective_device = hw["device"]
        else:
            self.effective_device = self.system.device.lower()
            
        # Mode resolution
        if self.system.mode == "auto":
            # If CUDA is absent or VRAM is insufficient for 7B VLM, use fallback
            if self.effective_device != "cuda" or hw["gpu_vram_gb"] < 12.0:
                self.effective_mode = "fallback"
            else:
                self.effective_mode = "real"
        else:
            self.effective_mode = self.system.mode.lower()
            
        # Ensure output directory exists
        base_dir = Path(__file__).resolve().parent
        out_path = base_dir / self.paths.output_dir
        out_path.mkdir(parents=True, exist_ok=True)
        return self


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Loads configuration from YAML file or returns resolved default config."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        cfg = AppConfig()
        return cfg.resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sys_raw = raw.get("system", {})
    paths_raw = raw.get("paths", {})
    prep_raw = raw.get("preprocessing", {})
    clip_raw = raw.get("changeclip", {})
    chat_raw = raw.get("changechat", {})
    fb_raw = raw.get("fallback", {})

    cfg = AppConfig(
        system=SystemConfig(**sys_raw),
        paths=PathsConfig(**paths_raw),
        preprocessing=PreprocessingConfig(**prep_raw),
        changeclip=ChangeCLIPConfig(**clip_raw),
        changechat=ChangeChatConfig(**chat_raw),
        fallback=FallbackConfig(**fb_raw),
    )
    return cfg.resolve()
