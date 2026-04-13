#!/usr/bin/env bash
set -euo pipefail

SAM_VARIANT="${1:-vit_b}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints"

mkdir -p "${CHECKPOINT_DIR}"

download_file() {
  local url="$1"
  local destination="$2"

  if [ -s "${destination}" ]; then
    echo "Skip existing file: ${destination}"
    return 0
  fi

  if [ -f "${destination}" ]; then
    echo "Remove incomplete file: ${destination}"
    rm -f "${destination}"
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -O "${destination}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${destination}"
  else
    echo "Neither wget nor curl is available."
    exit 1
  fi
}

GROUNDINGDINO_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
GROUNDINGDINO_DEST="${CHECKPOINT_DIR}/groundingdino_swint_ogc.pth"

case "${SAM_VARIANT}" in
  vit_b)
    SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    SAM_DEST="${CHECKPOINT_DIR}/sam_vit_b_01ec64.pth"
    ;;
  vit_l)
    SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    SAM_DEST="${CHECKPOINT_DIR}/sam_vit_l_0b3195.pth"
    ;;
  vit_h)
    SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    SAM_DEST="${CHECKPOINT_DIR}/sam_vit_h_4b8939.pth"
    ;;
  mobile_sam)
    SAM_URL="https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    SAM_DEST="${CHECKPOINT_DIR}/mobile_sam.pt"
    ;;
  sam2_tiny)
    SAM_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    SAM_DEST="${CHECKPOINT_DIR}/sam2.1_hiera_tiny.pt"
    ;;
  *)
    echo "Unsupported SAM variant: ${SAM_VARIANT}"
    echo "Use one of: vit_b, vit_l, vit_h, mobile_sam, sam2_tiny"
    exit 1
    ;;
esac

download_file "${GROUNDINGDINO_URL}" "${GROUNDINGDINO_DEST}"
download_file "${SAM_URL}" "${SAM_DEST}"

echo
echo "Weights downloaded into ${CHECKPOINT_DIR}"
