# RTX Pro 6000 Migration Notes

## Recommended Transfer Scope

Copy these directories to the target machine:

- `app/`
- `scripts/`
- `checkpoints/`
- `external/GroundingDINO/`
- `external/SAM2/`
- `external/segment-anything/`
- `external/MobileSAM/`

Optional:

- `inputs/` for smoke tests
- `outputs/` only if you want to keep old run records

Do not rely on copying an old compiled `GroundingDINO` extension between machines. Rebuild it on the target machine.

## Recommended Bring-Up Order

1. Create or prepare the target conda environment.
2. Install PyTorch for CUDA on the target machine.
3. Run `bash scripts/setup_project.sh SAM_DINO cu121`.
4. Run `bash scripts/download_weights.sh vit_b` if checkpoints were not copied.
5. Rebuild the GroundingDINO CUDA extension on the target machine:

```bash
bash scripts/rebuild_groundingdino_ext.sh SAM_DINO
```

6. Run the environment check:

```bash
bash scripts/verify_install.sh SAM_DINO
```

## CUDA / Driver Notes

- Keep the project on `--device cuda` for realtime inference.
- `GroundingDINO` and official `SAM2VideoPredictor` are now both configured for GPU-resident realtime execution.
- The target machine driver must be new enough for the selected PyTorch CUDA build.
- If the target machine uses a newer GPU architecture, rebuilding `GroundingDINO` is mandatory.
- The setup and rebuild scripts now try to append the detected GPU compute capability to `TORCH_CUDA_ARCH_LIST`, which helps when moving to a newer card.

## RealSense Notes

- This project uses the official `pyrealsense2` path for D435 realtime input.
- On Linux, make sure the target machine has `librealsense` and `pyrealsense2` available in the target environment.
- The realtime entrypoint already preloads `pyrealsense2` before Torch when `--source realsense` is used. Keep that import order unchanged.

## Fast Validation Commands

Realtime video smoke test:

```bash
conda run -n SAM_DINO python app/run_grounded_sam_realtime.py \
  --source inputs/ManyCameras.mp4 \
  --text "metal part . square metal part ." \
  --segmenter-backend sam2 \
  --sam2-config external/SAM2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --sam2-checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --device cuda \
  --box-threshold 0.22 \
  --text-threshold 0.18 \
  --min-box-area-ratio 0.0005 \
  --max-box-area-ratio 0.02 \
  --detection-interval 6 \
  --candidate-min-aspect-ratio 0.70 \
  --candidate-max-aspect-ratio 1.70 \
  --roi 0.12,0.04,0.90,0.93 \
  --auto-lock-best \
  --no-display \
  --max-frames 12 \
  --output-dir outputs/realtime_gpu_smoke
```

RealSense D435 realtime launch:

```bash
bash scripts/run_realsense_realtime.sh --auto-lock-best
```

## Practical Tuning Suggestions On Larger VRAM

- Try `--detection-interval 4` if reacquisition is still too slow.
- Try `--detection-max-side 960` or `1024` if small parts are missed.
- Keep `sam2.1_hiera_tiny` for lowest latency first; only move to larger SAM2 weights if latency remains acceptable.
- If the scene is fixed, keep `--roi` enabled to reduce false detections and improve recovery speed.
