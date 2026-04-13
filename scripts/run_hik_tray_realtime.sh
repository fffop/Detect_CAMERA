#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-SAM_DINO}"
SOURCE="${ROOT_DIR}/inputs/ManyCameras.mp4"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SAM2_VIDEO_MAX_FRAMES="${SAM2_VIDEO_MAX_FRAMES:-32}"
if [ "${1:-}" != "" ] && [[ "${1}" != --* ]]; then
  SOURCE="${1}"
  shift
fi

cd "${ROOT_DIR}"

conda run -n "${ENV_NAME}" env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" python app/run_grounded_sam_realtime.py \
  --source "${SOURCE}" \
  --device cuda \
  --text "metal part . square metal part ." \
  --segmenter-backend sam2_video \
  --sam2-config external/SAM2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --sam2-checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-video-max-frames "${SAM2_VIDEO_MAX_FRAMES}" \
  --box-threshold 0.18 \
  --text-threshold 0.12 \
  --min-box-area-ratio 0.0001 \
  --max-box-area-ratio 0.08 \
  --detection-interval 6 \
  --segmentation-interval 2 \
  --candidate-min-aspect-ratio 0.40 \
  --candidate-max-aspect-ratio 2.50 \
  "$@"
