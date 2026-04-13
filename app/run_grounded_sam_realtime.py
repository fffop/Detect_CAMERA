from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _should_preload_realsense(argv: list[str]) -> bool:
    normalized = [item.strip().lower() for item in argv[1:]]
    for index, item in enumerate(normalized):
        if item == "realsense":
            return True
        if item.startswith("--source=") and item.split("=", 1)[1] == "realsense":
            return True
        if item == "--source" and index + 1 < len(normalized) and normalized[index + 1] == "realsense":
            return True
    return False


if _should_preload_realsense(sys.argv):
    # In this environment, importing the RealSense SDK before torch/GroundingDINO
    # avoids a later std::bad_alloc during realtime CUDA inference.
    import pyrealsense2  # noqa: F401

from app.grounded_sam_core import (  # noqa: E402
    DEFAULT_GDINO_CHECKPOINT,
    DEFAULT_GDINO_CONFIG,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG_PATH,
    choose_device,
    ensure_exists,
    load_realtime_detector,
    load_realtime_models,
    load_sam2_video_tracker,
    write_json,
)
from app.realtime_pipeline import (  # noqa: E402
    FrameProcessResult,
    RealtimeGroundedPipeline,
    RealtimePipelineConfig,
)


class MouseState:
    def __init__(self) -> None:
        self.pending_click: tuple[int, int] | None = None


class RealSenseCapture:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        serial: str | None = None,
    ) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._profile = None
        if serial:
            self._config.enable_device(serial)
        self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._profile = self._pipeline.start(self._config)
        device = self._profile.get_device()
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size, 1)
        color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._width = color_stream.width()
        self._height = color_stream.height()
        self._fps = float(color_stream.fps())
        self._frame_index = 0

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            return False, None
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        frame_bgr = np.asarray(color_frame.get_data(), dtype=np.uint8).copy(order="C")
        del color_frame
        del frames
        self._frame_index += 1
        return True, frame_bgr

    def release(self) -> None:
        try:
            self._pipeline.stop()
        except RuntimeError:
            pass
        self._profile = None
        self._config = None
        self._pipeline = None

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self._frame_index)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        del prop_id, value
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime GroundingDINO candidate detection and single-target segmentation.")
    parser.add_argument("--source", default="0", help="Camera index, video path, or stream URL.")
    parser.add_argument("--realsense-width", type=int, default=1280, help="RealSense color stream width.")
    parser.add_argument("--realsense-height", type=int, default=720, help="RealSense color stream height.")
    parser.add_argument("--realsense-fps", type=int, default=30, help="RealSense color stream FPS.")
    parser.add_argument("--realsense-serial", default=None, help="Optional RealSense serial number when multiple devices are connected.")
    parser.add_argument("--text", required=True, help='Caption prompt, e.g. "camera module . lens module ."')
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "realtime_run"), help="Output directory.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gdino-config", default=str(DEFAULT_GDINO_CONFIG))
    parser.add_argument("--gdino-checkpoint", default=str(DEFAULT_GDINO_CHECKPOINT))
    parser.add_argument("--box-threshold", type=float, default=0.40)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-box-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.08)
    parser.add_argument("--detection-interval", type=int, default=5, help="Run GroundingDINO every N frames.")
    parser.add_argument("--segmentation-interval", type=int, default=1, help="Refresh segmentation every N frames while tracking.")
    parser.add_argument("--detection-max-side", type=int, default=800, help="Resize the long side before detection.")
    parser.add_argument("--candidate-min-aspect-ratio", type=float, default=0.4)
    parser.add_argument("--candidate-max-aspect-ratio", type=float, default=2.5)
    parser.add_argument(
        "--segmenter-backend",
        choices=("sam2", "sam2_video", "auto_fast", "grabcut", "sam", "mobile_sam"),
        default="sam2_video",
    )
    parser.add_argument("--sam-variant", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--mobile-sam-checkpoint", default=None)
    parser.add_argument("--sam2-config", default=str(DEFAULT_SAM2_CONFIG_PATH))
    parser.add_argument("--sam2-checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT))
    parser.add_argument(
        "--sam2-video-max-frames",
        type=int,
        default=32,
        help="Bounded-memory SAM2 video window. The tracker is reinitialized after this many frames to prevent OOM.",
    )
    parser.add_argument("--grabcut-iterations", type=int, default=1)
    parser.add_argument("--box-smoothing-alpha", type=float, default=0.70)
    parser.add_argument("--mask-smoothing-alpha", type=float, default=0.50)
    parser.add_argument("--template-search-margin-ratio", type=float, default=0.70)
    parser.add_argument("--template-match-threshold", type=float, default=0.18)
    parser.add_argument("--redetect-after-misses", type=int, default=5)
    parser.add_argument(
        "--manual-lock-reacquire-after-misses",
        type=int,
        default=3,
        help="After a manual click/selection, require this many consecutive misses before detection can reassign the target.",
    )
    parser.add_argument("--segmentation-expand-ratio", type=float, default=1.25)
    parser.add_argument("--roi", default=None, help="Normalized ROI as x1,y1,x2,y2 in [0,1].")
    parser.add_argument("--save-video", action="store_true", help="Save the annotated stream as MP4.")
    parser.add_argument("--no-display", action="store_true", help="Run without creating an OpenCV window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 means no limit.")
    parser.add_argument("--start-frame", type=int, default=0, help="Seek to this frame index before processing.")
    parser.add_argument("--auto-lock-best", action="store_true", help="Lock the highest-score candidate automatically.")
    parser.add_argument("--auto-lock-index", type=int, default=0, help="Candidate index to auto-lock in no-display runs. 0 means highest-score candidate.")
    return parser.parse_args()


def parse_capture_source(source: str) -> int | str:
    stripped = source.strip()
    if stripped.isdigit():
        return int(stripped)
    if "://" not in stripped:
        ensure_exists(Path(stripped), "Provide a valid local video path or a reachable stream URL.")
    return stripped


def open_capture(args: argparse.Namespace):
    if args.source.strip().lower() == "realsense":
        return RealSenseCapture(
            width=args.realsense_width,
            height=args.realsense_height,
            fps=args.realsense_fps,
            serial=args.realsense_serial,
        )
    return cv2.VideoCapture(parse_capture_source(args.source))


def parse_normalized_roi(raw_value: str | None) -> tuple[float, float, float, float] | None:
    if raw_value is None or raw_value.strip() == "":
        return None
    parts = [float(item.strip()) for item in raw_value.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be four comma-separated floats like 0.1,0.2,0.9,0.8")
    x1, y1, x2, y2 = parts
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("--roi values must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return x1, y1, x2, y2


def make_result_after_manual_lock(
    result: FrameProcessResult,
    pipeline: RealtimeGroundedPipeline,
    frame_index: int,
    message: str,
) -> FrameProcessResult:
    return FrameProcessResult(
        frame_index=frame_index,
        detection_ran=result.detection_ran,
        candidates=pipeline.last_candidates,
        locked_target=pipeline.target,
        state="locked" if pipeline.target is not None else result.state,
        message=message,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    video_tracker = None
    if args.segmenter_backend == "sam2_video":
        gdino_model = load_realtime_detector(
            device=device,
            gdino_config=args.gdino_config,
            gdino_checkpoint=args.gdino_checkpoint,
        )
        segmenter = None
        video_tracker = load_sam2_video_tracker(
            device=device,
            sam2_config=args.sam2_config,
            sam2_checkpoint=args.sam2_checkpoint,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            max_frames_per_track=max(8, args.sam2_video_max_frames),
        )
    else:
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
    pipeline = RealtimeGroundedPipeline(
        gdino_model=gdino_model,
        segmenter=segmenter,
        device=device,
        video_tracker=video_tracker,
        config=RealtimePipelineConfig(
            text_prompt=args.text,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            min_box_area_ratio=args.min_box_area_ratio,
            max_box_area_ratio=args.max_box_area_ratio,
            detection_interval=max(1, args.detection_interval),
            segmentation_interval=max(1, args.segmentation_interval),
            detection_max_side=max(0, args.detection_max_side),
            box_smoothing_alpha=args.box_smoothing_alpha,
            mask_smoothing_alpha=args.mask_smoothing_alpha,
            template_search_margin_ratio=args.template_search_margin_ratio,
            template_match_threshold=args.template_match_threshold,
            redetect_after_misses=max(1, args.redetect_after_misses),
            manual_lock_reacquire_after_misses=max(1, args.manual_lock_reacquire_after_misses),
            segmentation_expand_ratio=max(1.0, args.segmentation_expand_ratio),
            candidate_min_aspect_ratio=args.candidate_min_aspect_ratio,
            candidate_max_aspect_ratio=args.candidate_max_aspect_ratio,
            roi=parse_normalized_roi(args.roi),
        ),
    )

    capture = open_capture(args)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")
    if args.start_frame > 0 and args.source.strip().lower() != "realsense":
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(
            str(output_dir / "grounded_sam_realtime.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to create output video writer.")

    window_name = "GroundedSAM Realtime"
    mouse_state = MouseState()
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        def _mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
            del flags, param
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_state.pending_click = (x, y)

        cv2.setMouseCallback(window_name, _mouse_callback)

    summary = {
        "source": args.source,
        "text_prompt": args.text,
        "device": device,
        "segmenter_backend": video_tracker.backend_name if video_tracker is not None else segmenter.backend_name,
        "frames": [],
    }

    frame_index = max(0, args.start_frame)
    processed_frames = 0
    start_time = time.perf_counter()
    force_detect_next = False
    last_frame_rgb = None
    last_result = None

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            last_frame_rgb = frame_rgb
            result = pipeline.process_frame(
                frame_rgb=frame_rgb,
                frame_index=frame_index,
                force_detect=force_detect_next,
            )
            force_detect_next = False

            if mouse_state.pending_click is not None:
                pipeline.refresh_candidates(frame_rgb, frame_index)
                if pipeline.lock_target_from_point(mouse_state.pending_click, frame_rgb, frame_index):
                    result = make_result_after_manual_lock(result, pipeline, frame_index, "locked by click")
                mouse_state.pending_click = None

            if args.auto_lock_best and pipeline.target is None and result.candidates:
                lock_index = max(0, args.auto_lock_index)
                if lock_index >= len(result.candidates):
                    lock_index = 0
                if pipeline.lock_target_by_index(lock_index, frame_rgb, frame_index):
                    result = make_result_after_manual_lock(result, pipeline, frame_index, f"locked candidate {lock_index + 1}")

            annotated_bgr = pipeline.annotate_frame(frame_rgb, result)
            if writer is not None:
                writer.write(annotated_bgr)

            if not args.no_display:
                cv2.imshow(window_name, annotated_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    force_detect_next = True
                if key == ord("r"):
                    pipeline.reset_target()
                    result = make_result_after_manual_lock(result, pipeline, frame_index, "target reset")
                if ord("1") <= key <= ord("9") and last_frame_rgb is not None:
                    candidate_index = key - ord("1")
                    pipeline.refresh_candidates(last_frame_rgb, frame_index)
                    if pipeline.lock_target_by_index(
                        candidate_index,
                        last_frame_rgb,
                        frame_index,
                        manual_lock=True,
                    ):
                        result = make_result_after_manual_lock(result, pipeline, frame_index, f"locked candidate {candidate_index + 1}")

            summary["frames"].append(pipeline.frame_record(result))
            last_result = result
            frame_index += 1
            processed_frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    avg_fps = processed_frames / elapsed if elapsed > 0 else 0.0
    summary["processed_frames"] = processed_frames
    summary["elapsed_sec"] = round(elapsed, 3)
    summary["avg_fps"] = round(avg_fps, 3)
    if last_result is not None:
        summary["final_state"] = last_result.state
        summary["final_message"] = last_result.message

    write_json(output_dir / "result.json", summary)
    print(
        f"Processed {processed_frames} frame(s) in {elapsed:.2f}s. "
        f"Average throughput: {avg_fps:.2f} frame/s. "
        f"Segmenter backend: {video_tracker.backend_name if video_tracker is not None else segmenter.backend_name}."
    )


if __name__ == "__main__":
    main()
