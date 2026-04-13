#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-SAM_DINO}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-}"

if [ -z "${CONDA_BIN}" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [ -x "${HOME}/anaconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/anaconda3/bin/conda"
  elif [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
  else
    echo "conda not found."
    exit 1
  fi
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" env PROJECT_ROOT="${ROOT_DIR}" python - <<'PY'
import importlib.util
import os
from pathlib import Path
import sys

root = Path(os.environ["PROJECT_ROOT"]).resolve()
checks = {
    "torch": bool(importlib.util.find_spec("torch")),
    "torchvision": bool(importlib.util.find_spec("torchvision")),
    "cv2": bool(importlib.util.find_spec("cv2")),
    "PIL": bool(importlib.util.find_spec("PIL")),
    "hydra": bool(importlib.util.find_spec("hydra")),
    "iopath": bool(importlib.util.find_spec("iopath")),
    "groundingdino": bool(importlib.util.find_spec("groundingdino")),
    "segment_anything": bool(importlib.util.find_spec("segment_anything")),
    "sam2": bool(importlib.util.find_spec("sam2")),
    "pyrealsense2": bool(importlib.util.find_spec("pyrealsense2")),
}

print("Python package checks:")
for key, value in checks.items():
    print(f"  {key}: {value}")

try:
    import torch

    print("\nTorch runtime:")
    print(f"  version: {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    print(f"  cuda version: {getattr(torch.version, 'cuda', None)}")
    print(f"  device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            capability = ".".join(str(v) for v in torch.cuda.get_device_capability(index))
            print(f"  gpu[{index}]: {torch.cuda.get_device_name(index)} (cc {capability})")
except Exception as exc:
    print("\nTorch runtime:")
    print(f"  unavailable ({exc})")

print("\nFilesystem checks:")
print(f"  Project root: {root}")
print(f"  GroundingDINO repo: {(root / 'external' / 'GroundingDINO').exists()}")
print(f"  segment-anything repo: {(root / 'external' / 'segment-anything').exists()}")
print(f"  SAM2 repo: {(root / 'external' / 'SAM2').exists()}")
print(f"  GroundingDINO checkpoint: {(root / 'checkpoints' / 'groundingdino_swint_ogc.pth').exists()}")
print(f"  SAM vit_b checkpoint: {(root / 'checkpoints' / 'sam_vit_b_01ec64.pth').exists()}")
print(f"  MobileSAM checkpoint: {(root / 'checkpoints' / 'mobile_sam.pt').exists()}")
print(f"  SAM2 checkpoint: {(root / 'checkpoints' / 'sam2.1_hiera_tiny.pt').exists()}")
print(f"  BERT text encoder: {(root / 'checkpoints' / 'bert-base-uncased').exists()}")

sys.path.insert(0, str(root))
try:
    import app.grounded_sam_core  # noqa: F401
    import app.run_grounded_sam_realtime  # noqa: F401
    import groundingdino._C  # noqa: F401
    import sam2  # noqa: F401
    print("\nExtension checks:")
    print("  GroundingDINO _C extension: True")
    print("  Realtime entry import: True")
    print("  SAM2 import: True")
except Exception as exc:
    print("\nExtension checks:")
    print(f"  Runtime import check: False ({exc})")
PY
