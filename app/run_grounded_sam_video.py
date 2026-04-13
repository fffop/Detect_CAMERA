from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import cv2
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.grounded_sam_core import (  # noqa: E402
    DEFAULT_GDINO_CHECKPOINT,
    DEFAULT_GDINO_CONFIG,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG_PATH,
    build_empty_result,
    choose_device,
    ensure_exists,
    image_to_bgr_array,
    infer_on_rgb_image,
    load_realtime_models,
    save_individual_masks,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GroundingDINO + SAM on a video.")
    parser.add_argument("--video", required=True, help="Path to an input video.")
    parser.add_argument("--text", required=True, help='Caption prompt, e.g. "person . dog ."')
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "video_run"), help="Output directory.")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="GroundingDINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")
    parser.add_argument("--segmenter-backend", choices=("sam2", "auto_fast", "grabcut", "sam", "mobile_sam"), default="sam2")
    parser.add_argument("--sam-variant", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--min-box-area-ratio", type=float, default=0.0, help="Keep boxes larger than this fraction of frame area.")
    parser.add_argument("--max-box-area-ratio", type=float, default=1.0, help="Keep boxes smaller than this fraction of frame area.")
    parser.add_argument("--gdino-config", default=str(DEFAULT_GDINO_CONFIG))
    parser.add_argument("--gdino-checkpoint", default=str(DEFAULT_GDINO_CHECKPOINT))
    parser.add_argument("--sam-checkpoint", default=None, help="Optional custom SAM checkpoint path.")
    parser.add_argument("--mobile-sam-checkpoint", default=None, help="Optional custom MobileSAM checkpoint path.")
    parser.add_argument("--sam2-config", default=str(DEFAULT_SAM2_CONFIG_PATH), help="SAM2 config path.")
    parser.add_argument("--sam2-checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT), help="SAM2 checkpoint path.")
    parser.add_argument("--grabcut-iterations", type=int, default=1, help="GrabCut iterations when using grabcut backend.")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after processing this many frames. 0 means no limit.")
    parser.add_argument("--save-frame-masks", action="store_true", help="Save masks for each processed frame.")
    parser.add_argument("--save-annotated-frames", action="store_true", help="Save each processed annotated frame as JPEG.")
    parser.add_argument("--hold-last-detection-frames", type=int, default=0, help="Reuse the previous detection result for this many consecutive missed frames.")
    parser.add_argument("--inference-max-side", type=int, default=0, help="Resize the long side before detection.")
    parser.add_argument("--start-frame", type=int, default=0, help="Seek to this frame index before processing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = ensure_exists(Path(args.video), "Provide a valid input video path.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1")

    device = choose_device(args.device)
    gdino_model, segmenter = load_realtime_models(
        device=device,
        gdino_config=args.gdino_config,
        gdino_checkpoint=args.gdino_checkpoint,
        segmenter_backend=args.segmenter_backend,
        sam_variant=args.sam_variant,
        sam_checkpoint=args.sam_checkpoint,
        mobile_sam_checkpoint=args.mobile_sam_checkpoint,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        grabcut_iterations=args.grabcut_iterations,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    if args.start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    input_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = input_fps / args.frame_step if args.frame_step > 1 else input_fps

    writer = cv2.VideoWriter(
        str(output_dir / "grounded_sam_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Failed to create output video writer.")

    frames_dir = output_dir / "frames"
    if args.save_frame_masks or args.save_annotated_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "video": str(video_path),
        "text_prompt": args.text,
        "device": device,
        "segmenter_backend": segmenter.backend_name,
        "input_fps": input_fps,
        "output_fps": output_fps,
        "frame_step": args.frame_step,
        "total_frames_in_file": total_frame_count,
        "processed_frames": [],
    }

    source_name = str(video_path)
    frame_index = max(0, args.start_frame)
    processed_count = 0
    last_detection_state = None
    hold_count = 0
    planned_frames = total_frame_count // args.frame_step + int(total_frame_count % args.frame_step != 0)
    if args.max_frames > 0:
        planned_frames = min(planned_frames, args.max_frames)
    start_time = time.perf_counter()
    progress = tqdm(total=planned_frames if planned_frames > 0 else None, desc="Processing video", unit="frame", dynamic_ncols=True)

    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break

            current_index = frame_index
            frame_index += 1

            if current_index % args.frame_step != 0:
                continue
            if args.max_frames > 0 and processed_count >= args.max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = infer_on_rgb_image(
                image_rgb=frame_rgb,
                text_prompt=args.text,
                gdino_model=gdino_model,
                predictor=segmenter,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=device,
                min_box_area_ratio=args.min_box_area_ratio,
                max_box_area_ratio=args.max_box_area_ratio,
                inference_max_side=args.inference_max_side,
            )

            frame_name = f"frame_{current_index:06d}"
            frame_record = build_empty_result(source_name, args.text, device)
            frame_record["frame_index"] = current_index
            frame_record["timestamp_sec"] = round(current_index / input_fps, 3)

            if result["detections"]:
                writer.write(image_to_bgr_array(result["result_image"]))
                frame_record["detections"] = result["detections"]
                frame_record.pop("message", None)
                last_detection_state = {
                    "result_image": result["result_image"].copy(),
                    "detections": copy.deepcopy(result["detections"]),
                }
                hold_count = 0

                if args.save_annotated_frames:
                    result["result_image"].save(frames_dir / f"{frame_name}.jpg")
                if args.save_frame_masks and result["masks"] is not None:
                    frame_mask_files = save_individual_masks(frames_dir, result["masks"], prefix=frame_name)
                    for detection, mask_file in zip(frame_record["detections"], frame_mask_files):
                        detection["mask_file"] = mask_file
            else:
                can_hold = (
                    args.hold_last_detection_frames > 0
                    and last_detection_state is not None
                    and hold_count < args.hold_last_detection_frames
                )
                if can_hold:
                    writer.write(image_to_bgr_array(last_detection_state["result_image"]))
                    frame_record["detections"] = copy.deepcopy(last_detection_state["detections"])
                    frame_record["filled_from_previous"] = True
                    frame_record["source_frame_for_fill"] = current_index - 1 - hold_count
                    frame_record.pop("message", None)
                    hold_count += 1
                    if args.save_annotated_frames:
                        last_detection_state["result_image"].save(frames_dir / f"{frame_name}.jpg")
                else:
                    writer.write(frame_bgr)
                    hold_count = 0
                    if args.save_annotated_frames:
                        result["result_image"].save(frames_dir / f"{frame_name}.jpg")

            summary["processed_frames"].append(frame_record)
            processed_count += 1
            progress.update(1)
            elapsed = time.perf_counter() - start_time
            if elapsed > 0:
                progress.set_postfix(frame=current_index, det=len(frame_record["detections"]), fps=f"{processed_count / elapsed:.2f}")
    finally:
        progress.close()
        capture.release()
        writer.release()

    write_json(output_dir / "result.json", summary)
    elapsed = time.perf_counter() - start_time
    avg_fps = processed_count / elapsed if elapsed > 0 else 0.0
    print(f"Finished processing {processed_count} frame(s) in {elapsed:.2f}s. Average throughput: {avg_fps:.2f} frame/s.")


if __name__ == "__main__":
    main()
