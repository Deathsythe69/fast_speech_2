import os
import shutil
import sys

try:
    import torch
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("GPU name: None")
        print("GPU VRAM: 0 GB")
except ImportError:
    print("Torch not installed")

total, used, free = shutil.disk_usage("/")
print(f"Free disk space: {free / 1e9:.2f} GB")

for cmd in ("mfa", "ffmpeg"):
    print(f"{cmd} on PATH: {shutil.which(cmd) is not None}")
