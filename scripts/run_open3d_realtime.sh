#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-SAM_DINO}"
TEXT_PROMPT="${TEXT_PROMPT:-metal part . square metal part .}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/open3d_realtime_run}"
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
  --text "${TEXT_PROMPT}"
  --output-dir "${OUTPUT_DIR}"
  --segmenter-backend "${SEGMENTER_BACKEND:-sam2_video}"
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
  --depth-min-m "${DEPTH_MIN_M:-0.10}"
  --depth-max-m "${DEPTH_MAX_M:-1.20}"
  --geometry-voxel-size-m "${GEOMETRY_VOXEL_SIZE_M:-0.0025}"
  --geometry-min-point-count "${GEOMETRY_MIN_POINT_COUNT:-120}"
  --geometry-dbscan-eps-m "${GEOMETRY_DBSCAN_EPS_M:-0.006}"
  --geometry-dbscan-min-points "${GEOMETRY_DBSCAN_MIN_POINTS:-20}"
  --geometry-axis-scale-m "${GEOMETRY_AXIS_SCALE_M:-0.03}"
  --auto-lock-best
  --auto-lock-index "${AUTO_LOCK_INDEX:-0}"
  --hide-candidates-after-lock
)

if [[ -n "${REALSENSE_SERIAL:-}" ]]; then
  ARGS+=(--realsense-serial "${REALSENSE_SERIAL}")
fi

if [[ -n "${ROI_VALUE}" ]]; then
  ARGS+=(--roi "${ROI_VALUE}")
fi

if [[ "${AUTO_LOCK_BEST}" == "1" ]]; then
  ARGS+=(--auto-lock-best --auto-lock-index "${AUTO_LOCK_INDEX}")
fi

conda run -n "${ENV_NAME}" env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" python app/run_open3d_realtime.py \
  "${ARGS[@]}" \
  "$@"
