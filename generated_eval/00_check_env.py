#!/usr/bin/env python
"""Check the local environment for generated video evaluation."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


def check_import(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "version unknown")
        print(f"[OK] import {module_name}: {version}")
    except Exception as exc:
        print(f"[FAIL] import {module_name}: {type(exc).__name__}: {exc}")


def check_torch() -> None:
    try:
        import torch

        print(f"[OK] import torch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
        else:
            print("GPU name: unavailable")
    except Exception as exc:
        print(f"[FAIL] import torch: {type(exc).__name__}: {exc}")
        print("CUDA check skipped because torch could not be imported.")


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/generated_eval_mplconfig")
    print("Generated video evaluation environment check")
    print(f"Python: {sys.version.replace(chr(10), ' ')}")
    print(f"Platform: {platform.platform()}")
    print(f"Current working directory: {Path.cwd()}")
    print("")

    check_torch()
    print("")

    for module_name in [
        "cv2",
        "PIL",
        "numpy",
        "transformers",
        "matplotlib",
        "sklearn",
    ]:
        check_import(module_name)


if __name__ == "__main__":
    main()
