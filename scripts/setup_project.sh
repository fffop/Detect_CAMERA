#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-SAM_DINO}"
TORCH_FLAVOR="${2:-cpu}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-$(printf '%s\n' "${PIP_INDEX_URL}" | awk -F/ '{print $3}')}"
PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
TORCH_INDEX_BASE="${TORCH_INDEX_BASE:-https://mirrors.aliyun.com/pytorch-wheels}"
GIT_MIRROR_PREFIX="${GIT_MIRROR_PREFIX:-}"
CONDA_BIN="${CONDA_BIN:-}"
if [ -z "${CONDA_BIN}" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [ -x "${HOME}/anaconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/anaconda3/bin/conda"
  elif [ -x "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
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

if [ -z "${CONDA_BIN}" ]; then
  echo "conda not found. Please install Anaconda or Miniconda first."
  exit 1
fi

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' does not exist."
  exit 1
fi

eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate "${ENV_NAME}"

pip_install() {
  python -m pip install \
    --default-timeout "${PIP_DEFAULT_TIMEOUT}" \
    -i "${PIP_INDEX_URL}" \
    --trusted-host "${PIP_TRUSTED_HOST}" \
    "$@"
}

pip_install_torch() {
  local torch_channel="$1"
  python -m pip install \
    --default-timeout "${PIP_DEFAULT_TIMEOUT}" \
    --upgrade \
    --force-reinstall \
    --no-cache-dir \
    --index-url "${TORCH_INDEX_BASE}/${torch_channel}" \
    --extra-index-url "${PIP_INDEX_URL}" \
    --trusted-host "${PIP_TRUSTED_HOST}" \
    torch torchvision
}

detect_torch_cuda_arch_list() {
  python - <<'PY'
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
  python - "$1" "$2" <<'PY'
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

clone_repo() {
  local repo_url="$1"
  local target_dir="$2"
  git clone --depth 1 "${GIT_MIRROR_PREFIX}${repo_url}" "${target_dir}"
}

python --version
echo "Using pip mirror: ${PIP_INDEX_URL}"
echo "Using torch index base: ${TORCH_INDEX_BASE}"
[ -n "${GIT_MIRROR_PREFIX}" ] && echo "Using git mirror prefix: ${GIT_MIRROR_PREFIX}"
[ -n "${CUDA_HOME}" ] && echo "Using CUDA_HOME: ${CUDA_HOME}"

pip_install --upgrade pip setuptools wheel

case "${TORCH_FLAVOR}" in
  cpu)
    pip_install_torch cpu
    ;;
  cu118)
    pip_install_torch cu118
    ;;
  cu121)
    pip_install_torch cu121
    ;;
  *)
    echo "Unsupported torch flavor: ${TORCH_FLAVOR}"
    echo "Use one of: cpu, cu118, cu121"
    exit 1
    ;;
esac

if [ "${TORCH_FLAVOR}" != "cpu" ]; then
  DETECTED_TORCH_ARCH_LIST="$(detect_torch_cuda_arch_list || true)"
  if [ -n "${DETECTED_TORCH_ARCH_LIST}" ]; then
    TORCH_CUDA_ARCH_LIST="$(merge_arch_lists "${TORCH_CUDA_ARCH_LIST}" "${DETECTED_TORCH_ARCH_LIST}")"
  fi
fi

pip_install \
  numpy \
  pillow \
  opencv-python \
  matplotlib \
  tqdm \
  requests \
  scipy \
  supervision \
  pycocotools \
  onnxruntime \
  addict \
  yapf \
  timm \
  "transformers>=4.30,<5" \
  ninja

# SAM2 is installed with --no-deps below so that we don't let its metadata
# override the project's torch build, but its runtime dependencies still need
# to be installed explicitly.
pip_install \
  "hydra-core>=1.3.2" \
  "iopath>=0.1.10"

mkdir -p "${ROOT_DIR}/external"

if [ ! -d "${ROOT_DIR}/external/GroundingDINO/.git" ]; then
  clone_repo https://github.com/IDEA-Research/GroundingDINO.git "${ROOT_DIR}/external/GroundingDINO"
fi

if [ ! -d "${ROOT_DIR}/external/segment-anything/.git" ]; then
  clone_repo https://github.com/facebookresearch/segment-anything.git "${ROOT_DIR}/external/segment-anything"
fi

if [ ! -d "${ROOT_DIR}/external/MobileSAM/.git" ]; then
  clone_repo https://github.com/ChaoningZhang/MobileSAM.git "${ROOT_DIR}/external/MobileSAM"
fi

if [ ! -d "${ROOT_DIR}/external/SAM2/.git" ]; then
  clone_repo https://github.com/facebookresearch/sam2.git "${ROOT_DIR}/external/SAM2"
fi

pip_install --no-build-isolation --no-deps -e "${ROOT_DIR}/external/GroundingDINO"
pip_install --no-deps -e "${ROOT_DIR}/external/segment-anything"
pip_install --no-deps -e "${ROOT_DIR}/external/MobileSAM"
pip_install --no-build-isolation --no-deps -e "${ROOT_DIR}/external/SAM2"

if [ -n "${CUDA_HOME}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
  echo "Building GroundingDINO with TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  (
    cd "${ROOT_DIR}/external/GroundingDINO"
    clean_groundingdino_build_artifacts
    CUDA_HOME="${CUDA_HOME}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
      python setup.py build_ext --inplace
  )
fi

echo
echo "Project setup finished in environment '${ENV_NAME}'."
echo "Next step: bash scripts/download_weights.sh vit_b"
