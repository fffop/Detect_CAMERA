#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-SAM_DINO}"
TEXT_PROMPT="${TEXT_PROMPT:-metal part . square metal part .}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/realsense_realtime_run}"
ROI_VALUE="${ROI:-}"
AUTO_LOCK_BEST="${AUTO_LOCK_BEST:-1}"
AUTO_LOCK_INDEX="${AUTO_LOCK_INDEX:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SAM2_VIDEO_MAX_FRAMES="${SAM2_VIDEO_MAX_FRAMES:-32}"

cd "${ROOT_DIR}"

ARGS=(
  --source realsense
  --device cuda
  --realsense-width "${REALSENSE_WIDTH:-1280}"
  --realsense-height "${REALSENSE_HEIGHT:-720}"
  --realsense-fps "${REALSENSE_FPS:-30}"
  --realsense-serial "${REALSENSE_SERIAL:-}"
  --text "${TEXT_PROMPT}"
  --output-dir "${OUTPUT_DIR}"
  --segmenter-backend sam2_video
  --sam2-config external/SAM2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
  --sam2-checkpoint checkpoints/sam2.1_hiera_tiny.pt
  --sam2-video-max-frames "${SAM2_VIDEO_MAX_FRAMES}"
  --box-threshold "${BOX_THRESHOLD:-0.18}"
  --text-threshold "${TEXT_THRESHOLD:-0.12}"
  --min-box-area-ratio "${MIN_BOX_AREA_RATIO:-0.0001}"
  --max-box-area-ratio "${MAX_BOX_AREA_RATIO:-0.08}"
  --detection-interval "${DETECTION_INTERVAL:-6}"
  --segmentation-interval "${SEGMENTATION_INTERVAL:-2}"
  --candidate-min-aspect-ratio "${CANDIDATE_MIN_ASPECT_RATIO:-0.40}"
  --candidate-max-aspect-ratio "${CANDIDATE_MAX_ASPECT_RATIO:-2.50}"
)

if [[ -n "${ROI_VALUE}" ]]; then
  ARGS+=(--roi "${ROI_VALUE}")
fi

if [[ "${AUTO_LOCK_BEST}" == "1" ]]; then
  ARGS+=(--auto-lock-best --auto-lock-index "${AUTO_LOCK_INDEX}")
fi

conda run -n "${ENV_NAME}" env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" python app/run_grounded_sam_realtime.py \
  "${ARGS[@]}" \
  "$@"
