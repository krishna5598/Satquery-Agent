"""
Hardware and Environment Detection Utility for SatQuery AI Change Analysis.

Inspects CPU, RAM, GPU, CUDA availability, PyTorch version, and determines
the recommended operating mode (GPU / CPU / FALLBACK).
"""

from __future__ import annotations
import os
import platform
import sys
from typing import Dict, Any

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


def detect_hardware() -> Dict[str, Any]:
    """Inspects machine specs and returns a comprehensive capability report."""
    os_name = f"{platform.system()} {platform.release()} ({platform.version()})"
    python_ver = sys.version.split()[0]
    
    # CPU & RAM
    cpu_name = platform.processor() or "Unknown CPU"
    cpu_count_logical = os.cpu_count() or 1
    cpu_count_physical = psutil.cpu_count(logical=False) if psutil else cpu_count_logical // 2
    
    total_ram_gb = 0.0
    free_ram_gb = 0.0
    if psutil:
        mem = psutil.virtual_memory()
        total_ram_gb = round(mem.total / (1024 ** 3), 2)
        free_ram_gb = round(mem.available / (1024 ** 3), 2)
    
    # PyTorch & Device Detection
    torch_version = torch.__version__ if torch else "Not Installed"
    cuda_available = torch.cuda.is_available() if torch else False
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available() if torch else False
    
    gpu_name = "None"
    gpu_vram_gb = 0.0
    
    if cuda_available and cuda_device_count > 0:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_properties = torch.cuda.get_device_properties(0)
        gpu_vram_gb = round(gpu_properties.total_memory / (1024 ** 3), 2)
    elif platform.system() == "Windows":
        # Check for integrated or discrete GPU via wmic/cim if torch doesn't see CUDA
        try:
            import subprocess
            cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
            res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
            names = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            if names:
                gpu_name = ", ".join(names)
        except Exception:
            gpu_name = "Integrated / Unknown"

    # Select Optimal Device & Execution Mode
    if cuda_available and gpu_vram_gb >= 12.0:
        device = "cuda"
        recommended_mode = "GPU"
        note = "Adequate GPU VRAM detected. Real model inference supported."
    elif cuda_available and gpu_vram_gb < 12.0:
        device = "cuda"
        recommended_mode = "GPU (Limited VRAM)"
        note = "Limited GPU VRAM detected (<12GB). Quantization or CPU offload recommended for large models."
    elif mps_available:
        device = "mps"
        recommended_mode = "MPS (Apple Silicon)"
        note = "Apple Metal Performance Shaders available."
    elif total_ram_gb >= 16.0:
        device = "cpu"
        recommended_mode = "CPU"
        note = "CPU execution active with adequate system RAM."
    else:
        device = "cpu"
        recommended_mode = "FALLBACK / CPU"
        note = (
            f"Host machine has {total_ram_gb}GB RAM and no CUDA acceleration. "
            "Deterministic fallback is strongly recommended for responsive demo & evaluation."
        )

    return {
        "os": os_name,
        "python": python_ver,
        "pytorch": torch_version,
        "cpu": cpu_name,
        "cpu_cores_physical": cpu_count_physical,
        "cpu_cores_logical": cpu_count_logical,
        "total_ram_gb": total_ram_gb,
        "free_ram_gb": free_ram_gb,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "gpu": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "device": device,
        "recommended_mode": recommended_mode,
        "note": note,
    }


def print_hardware_report() -> None:
    """Prints a clean, formatted report matching the user-specified template."""
    info = detect_hardware()
    print("=" * 60)
    print(" SatQuery AI - Hardware & Environment Inspection Report")
    print("=" * 60)
    print(f" Operating System  : {info['os']}")
    print(f" Python Version    : {info['python']}")
    print(f" PyTorch Version   : {info['pytorch']}")
    print(f" CPU               : {info['cpu']} ({info['cpu_cores_logical']} threads)")
    print(f" System RAM        : {info['total_ram_gb']} GB total ({info['free_ram_gb']} GB available)")
    print(f" CUDA Available    : {'YES' if info['cuda_available'] else 'NO'}")
    print(f" GPU Name          : {info['gpu']}")
    print(f" GPU VRAM          : {info['gpu_vram_gb']} GB")
    print(f" Selected Device   : {info['device']}")
    print(f" Recommended Mode  : {info['recommended_mode']}")
    print("-" * 60)
    print(f" Assessment        : {info['note']}")
    print("=" * 60)


if __name__ == "__main__":
    print_hardware_report()
