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

detect_cuda_home_default() {
  local direct_path="/usr/local/cuda"
  local latest_versioned=""

  if [ -d /usr/local ]; then
    latest_versioned="$(
      find /usr/local -maxdepth 1 -mindepth 1 -type d -name 'cuda-*' 2>/dev/null \
        | sort -V \
        | tail -n 1
    )"
  fi

  if [ -n "${latest_versioned}" ] && [ -x "${latest_versioned}/bin/nvcc" ]; then
    printf '%s\n' "${latest_versioned}"
    return
  fi

  if [ -x "${direct_path}/bin/nvcc" ]; then
    printf '%s\n' "${direct_path}"
    return
  fi

  printf '\n'
}

CUDA_HOME_DEFAULT="$(detect_cuda_home_default)"

CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_DEFAULT}}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0+PTX}"

detect_torch_cuda_arch_list() {
  "${CONDA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
    raise SystemExit(0)

capabilities = []
for index in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(index)
    arch = f"{major}.{minor}"
    if arch not in capabilities:
        capabilities.append(arch)

print(";".join(capabilities))
PY
}

merge_arch_lists() {
  "${CONDA_BIN}" run -n "${ENV_NAME}" python - "$1" "$2" <<'PY'
import sys

merged = []
for group in sys.argv[1:]:
    for item in group.split(";"):
        item = item.strip()
        if item and item not in merged:
            merged.append(item)

print(";".join(merged))
PY
}

clean_groundingdino_build_artifacts() {
  local build_dir="${ROOT_DIR}/external/GroundingDINO/build"
  if [ -d "${build_dir}" ]; then
    rm -rf "${build_dir}"
  fi

  find "${ROOT_DIR}/external/GroundingDINO/groundingdino" -maxdepth 1 -type f \
    \( -name '_C*.so' -o -name '_C*.pyd' \) -delete
}

if [ -z "${CUDA_HOME}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  echo "CUDA toolkit with nvcc not found. Set CUDA_HOME manually before running this script."
  exit 1
fi

DETECTED_TORCH_ARCH_LIST="$(detect_torch_cuda_arch_list || true)"
if [ -n "${DETECTED_TORCH_ARCH_LIST}" ]; then
  TORCH_CUDA_ARCH_LIST="$(merge_arch_lists "${TORCH_CUDA_ARCH_LIST}" "${DETECTED_TORCH_ARCH_LIST}")"
fi

echo "Rebuilding GroundingDINO extension with CUDA_HOME=${CUDA_HOME}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

cd "${ROOT_DIR}/external/GroundingDINO"
echo "Cleaning previous GroundingDINO build artifacts..."
clean_groundingdino_build_artifacts
CUDA_HOME="${CUDA_HOME}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  "${CONDA_BIN}" run -n "${ENV_NAME}" python setup.py build_ext --inplace

echo
echo "Rebuild complete."
