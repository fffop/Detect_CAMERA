from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.grounded_sam_core import (
    BaseBoxSegmenter,
    blend_binary_masks,
    box_xyxy_area,
    box_xyxy_center,
    box_xyxy_iou,
    box_xyxy_size,
    clamp_box_xyxy,
    expand_box_xyxy,
    infer_on_rgb_image,
    make_box_mask,
    mask_to_box_xyxy,
    normalized_roi_to_xyxy,
    smooth_box_xyxy,
)


@dataclass
class RealtimePipelineConfig:
    text_prompt: str
    box_threshold: float = 0.40
    text_threshold: float = 0.25
    min_box_area_ratio: float = 0.0
    max_box_area_ratio: float = 0.08
    dedupe_iou_threshold: float = 0.65
    detection_interval: int = 5
    segmentation_interval: int = 1
    detection_max_side: int = 800
    box_smoothing_alpha: float = 0.70
    mask_smoothing_alpha: float = 0.50
    template_search_margin_ratio: float = 0.70
    template_match_threshold: float = 0.18
    redetect_after_misses: int = 5
    segmentation_expand_ratio: float = 1.25
    manual_lock_reacquire_after_misses: int = 3
    candidate_min_aspect_ratio: float = 0.4
    candidate_max_aspect_ratio: float = 2.5
    roi: tuple[float, float, float, float] | None = None
    show_candidates: bool = True


@dataclass
class DetectionCandidate:
    index: int
    label: str
    score: float
    box_xyxy: list[float]
    rank_score: float = 0.0


@dataclass
class LockedTarget:
    label: str
    score: float
    raw_box_xyxy: list[float]
    smoothed_box_xyxy: list[float]
    mask: np.ndarray | None = None
    mask_prob: np.ndarray | None = None
    status: str = "locked"
    last_seen_frame: int = 0
    last_detection_frame: int = 0
    miss_count: int = 0
    tracking_score: float = 0.0
    is_manual_lock: bool = False


@dataclass
class FrameProcessResult:
    frame_index: int
    detection_ran: bool
    candidates: list[DetectionCandidate]
    locked_target: LockedTarget | None
    state: str
    message: str


class RealtimeGroundedPipeline:
    def __init__(
        self,
        gdino_model,
        segmenter: BaseBoxSegmenter,
        device: str,
        config: RealtimePipelineConfig,
        video_tracker=None,
    ):
        self.gdino_model = gdino_model
        self.segmenter = segmenter
        self.device = device
        self.config = config
        self.video_tracker = video_tracker
        self.target: LockedTarget | None = None
        self.last_candidates: list[DetectionCandidate] = []
        self.last_detection_frame = -1

    def reset_target(self) -> None:
        self.target = None
        if self.video_tracker is not None:
            self.video_tracker.reset()

    def refresh_candidates(self, frame_rgb: np.ndarray, frame_index: int) -> list[DetectionCandidate]:
        candidates = self.detect_candidates(frame_rgb)
        self.last_candidates = candidates
        self.last_detection_frame = frame_index
        return candidates

    def lock_target_by_index(
        self,
        candidate_index: int,
        frame_rgb: np.ndarray,
        frame_index: int,
        manual_lock: bool = False,
    ) -> bool:
        if candidate_index < 0 or candidate_index >= len(self.last_candidates):
            return False
        candidate = self.last_candidates[candidate_index]
        self.target = LockedTarget(
            label=candidate.label,
            score=candidate.score,
            raw_box_xyxy=candidate.box_xyxy.copy(),
            smoothed_box_xyxy=candidate.box_xyxy.copy(),
            last_seen_frame=frame_index,
            last_detection_frame=frame_index,
            is_manual_lock=manual_lock,
        )
        if self.video_tracker is not None:
            tracking_output = self.video_tracker.initialize(frame_rgb, candidate.box_xyxy)
            if tracking_output is not None:
                self._apply_tracking_output(
                    frame_rgb=frame_rgb,
                    frame_index=frame_index,
                    tracking_output=tracking_output,
                    score=candidate.score,
                    label=candidate.label,
                    source="det",
                )
                return True
        self._refresh_target_mask(frame_rgb, frame_index)
        return True

    def lock_target_from_point(
        self,
        point_xy: tuple[int, int],
        frame_rgb: np.ndarray,
        frame_index: int,
    ) -> bool:
        px, py = point_xy
        chosen_index = None
        for candidate in self.last_candidates:
            x1, y1, x2, y2 = candidate.box_xyxy
            if x1 <= px <= x2 and y1 <= py <= y2:
                chosen_index = candidate.index
                break
        if chosen_index is None and self.last_candidates:
            chosen_index = min(
                self.last_candidates,
                key=lambda item: _center_distance(item.box_xyxy, [px, py, px, py]),
            ).index
        if chosen_index is None:
            return False
        return self.lock_target_by_index(
            chosen_index,
            frame_rgb=frame_rgb,
            frame_index=frame_index,
            manual_lock=True,
        )

    def process_frame(self, frame_rgb: np.ndarray, frame_index: int, force_detect: bool = False) -> FrameProcessResult:
        detection_ran = False
        message = "searching"
        candidates = self.last_candidates

        if self._should_run_detection(frame_index, force_detect):
            candidates = self.refresh_candidates(frame_rgb, frame_index)
            detection_ran = True

        if self.target is None:
            state = "search"
            if candidates:
                message = f"{len(candidates)} candidate(s)"
            else:
                message = "no candidates"
            return FrameProcessResult(
                frame_index=frame_index,
                detection_ran=detection_ran,
                candidates=candidates,
                locked_target=None,
                state=state,
                message=message,
            )

        matched_candidate = None
        if detection_ran:
            matched_candidate = self._match_target_candidate(candidates)
        if self.video_tracker is not None:
            should_prefer_tracking = self.target.is_manual_lock and not self._manual_lock_can_reacquire()
            if should_prefer_tracking:
                tracking_output = self.video_tracker.track(frame_rgb)
                if tracking_output is not None and tracking_output["score"] >= 0.05:
                    self._apply_tracking_output(
                        frame_rgb=frame_rgb,
                        frame_index=frame_index,
                        tracking_output=tracking_output,
                        score=tracking_output["score"],
                        label=self.target.label,
                        source="track",
                    )
                    message = f"tracking ({tracking_output['score']:.2f})"
                else:
                    self._mark_target_lost()
                    message = f"lost ({self.target.miss_count})"
                    if detection_ran and self._manual_lock_can_reacquire() and matched_candidate is not None:
                        tracking_output = self.video_tracker.initialize(frame_rgb, matched_candidate.box_xyxy)
                        if tracking_output is not None:
                            self._apply_tracking_output(
                                frame_rgb=frame_rgb,
                                frame_index=frame_index,
                                tracking_output=tracking_output,
                                score=max(matched_candidate.score, tracking_output["score"]),
                                label=matched_candidate.label,
                                source="det",
                            )
                            message = f"relocked via detection ({matched_candidate.label})"
                        else:
                            self._mark_target_lost()
                            message = f"lost ({self.target.miss_count})"
            elif matched_candidate is not None:
                tracking_output = self.video_tracker.initialize(frame_rgb, matched_candidate.box_xyxy)
                if tracking_output is not None:
                    self._apply_tracking_output(
                        frame_rgb=frame_rgb,
                        frame_index=frame_index,
                        tracking_output=tracking_output,
                        score=max(matched_candidate.score, tracking_output["score"]),
                        label=matched_candidate.label,
                        source="det",
                    )
                    message = f"locked via detection ({matched_candidate.label})"
                else:
                    self._mark_target_lost()
                    message = f"lost ({self.target.miss_count})"
            else:
                tracking_output = self.video_tracker.track(frame_rgb)
                if tracking_output is not None and tracking_output["score"] >= 0.05:
                    self._apply_tracking_output(
                        frame_rgb=frame_rgb,
                        frame_index=frame_index,
                        tracking_output=tracking_output,
                        score=tracking_output["score"],
                        label=self.target.label,
                        source="track",
                    )
                    message = f"tracking ({tracking_output['score']:.2f})"
                else:
                    self._mark_target_lost()
                    message = f"lost ({self.target.miss_count})"
        else:
            if matched_candidate is not None:
                self._update_target_from_box(
                    frame_rgb=frame_rgb,
                    frame_index=frame_index,
                    box_xyxy=matched_candidate.box_xyxy,
                    score=matched_candidate.score,
                    label=matched_candidate.label,
                    detection_source="det",
                )
                message = f"locked via detection ({matched_candidate.label})"
            else:
                self._mark_target_lost()
                message = f"lost ({self.target.miss_count})"

        if self.target.miss_count > self.config.redetect_after_misses:
            self.target.status = "lost"

        return FrameProcessResult(
            frame_index=frame_index,
            detection_ran=detection_ran,
            candidates=candidates,
            locked_target=self.target,
                state=self.target.status,
                message=message,
        )

    def detect_candidates(self, frame_rgb: np.ndarray) -> list[DetectionCandidate]:
        result = infer_on_rgb_image(
            image_rgb=frame_rgb,
            text_prompt=self.config.text_prompt,
            gdino_model=self.gdino_model,
            predictor=None,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            device=self.device,
            min_box_area_ratio=self.config.min_box_area_ratio,
            max_box_area_ratio=self.config.max_box_area_ratio,
            dedupe_iou_threshold=self.config.dedupe_iou_threshold,
            apply_sam=False,
            inference_max_side=self.config.detection_max_side,
        )
        roi_box = normalized_roi_to_xyxy(self.config.roi, frame_rgb.shape)
        filtered: list[DetectionCandidate] = []
        for detection in result["detections"]:
            box_xyxy = [float(v) for v in detection["box_xyxy"]]
            if not self._passes_geometry_filters(box_xyxy, frame_rgb.shape):
                continue
            if roi_box is not None and not _box_center_in_box(box_xyxy, roi_box):
                continue
            filtered.append(
                DetectionCandidate(
                    index=len(filtered),
                    label=str(detection["label"]),
                    score=float(detection["score"]),
                    box_xyxy=box_xyxy,
                )
            )
        filtered.sort(key=lambda item: item.score, reverse=True)
        for idx, candidate in enumerate(filtered):
            candidate.index = idx
        return filtered

    def annotate_frame(self, frame_rgb: np.ndarray, result: FrameProcessResult) -> np.ndarray:
        annotated = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
        roi_box = normalized_roi_to_xyxy(self.config.roi, frame_rgb.shape)
        if roi_box is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in roi_box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)

        if result.locked_target is not None and result.locked_target.mask is not None:
            mask = result.locked_target.mask.astype(bool)
            overlay = annotated.copy()
            overlay[mask] = (0, 180, 0)
            annotated = cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0.0)

        if self.config.show_candidates:
            for candidate in result.candidates:
                x1, y1, x2, y2 = [int(round(v)) for v in candidate.box_xyxy]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 215, 255), 1)
                cv2.putText(
                    annotated,
                    f"{candidate.index + 1}:{candidate.label} {candidate.score:.2f}",
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 215, 255),
                    1,
                    cv2.LINE_AA,
                )

        if result.locked_target is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in result.locked_target.smoothed_box_xyxy]
            lock_prefix = "MANUAL" if result.locked_target.is_manual_lock else "LOCK"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 220, 80), 2)
            cv2.putText(
                annotated,
                f"{lock_prefix} {result.locked_target.label} {result.state}",
                (x1, max(22, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 220, 80),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            annotated,
            result.message,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def frame_record(self, result: FrameProcessResult) -> dict:
        payload = {
            "frame_index": result.frame_index,
            "detection_ran": result.detection_ran,
            "state": result.state,
            "message": result.message,
            "candidates": [
                {
                    "index": candidate.index,
                    "label": candidate.label,
                    "score": round(candidate.score, 6),
                    "rank_score": round(candidate.rank_score, 6),
                    "box_xyxy": [round(v, 2) for v in candidate.box_xyxy],
                }
                for candidate in result.candidates
            ],
        }
        if result.locked_target is not None:
            mask_box = mask_to_box_xyxy(result.locked_target.mask)
            payload["locked_target"] = {
                "label": result.locked_target.label,
                "score": round(result.locked_target.score, 6),
                "tracking_score": round(result.locked_target.tracking_score, 6),
                "miss_count": result.locked_target.miss_count,
                "is_manual_lock": result.locked_target.is_manual_lock,
                "box_xyxy": [round(v, 2) for v in result.locked_target.smoothed_box_xyxy],
                "mask_area": int(result.locked_target.mask.sum()) if result.locked_target.mask is not None else 0,
                "mask_box_xyxy": [round(v, 2) for v in mask_box] if mask_box is not None else None,
            }
        return payload

    def _should_run_detection(self, frame_index: int, force_detect: bool) -> bool:
        if force_detect or self.last_detection_frame < 0:
            return True
        if self.target is None:
            return (frame_index - self.last_detection_frame) >= self.config.detection_interval
        if self.target.miss_count > 0:
            return True
        return (frame_index - self.last_detection_frame) >= self.config.detection_interval

    def _manual_lock_can_reacquire(self) -> bool:
        if self.target is None or not self.target.is_manual_lock:
            return True
        return self.target.miss_count >= max(1, self.config.manual_lock_reacquire_after_misses)

    def _match_target_candidate(self, candidates: list[DetectionCandidate]) -> DetectionCandidate | None:
        if self.target is None:
            return None
        gated_candidates = [
            candidate
            for candidate in candidates
            if (
                box_xyxy_iou(candidate.box_xyxy, self.target.smoothed_box_xyxy) >= 0.05
                or _normalized_center_distance(candidate.box_xyxy, self.target.smoothed_box_xyxy) <= 0.45
            )
        ]
        if gated_candidates:
            candidates = gated_candidates
        scored_candidates: list[tuple[float, DetectionCandidate]] = []
        for candidate in candidates:
            score = self._candidate_rank_score(candidate, self.target.smoothed_box_xyxy)
            candidate.rank_score = score
            scored_candidates.append((score, candidate))
        if not scored_candidates:
            return None
        same_label_candidates = [
            (score, candidate)
            for score, candidate in scored_candidates
            if candidate.label == self.target.label
        ]
        if same_label_candidates:
            scored_candidates = same_label_candidates
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate = scored_candidates[0]
        if best_score < 0.20:
            return None
        return best_candidate

    def _candidate_rank_score(
        self,
        candidate: DetectionCandidate,
        reference_box_xyxy: list[float],
    ) -> float:
        iou_score = box_xyxy_iou(candidate.box_xyxy, reference_box_xyxy)
        distance_score = 1.0 - _normalized_center_distance(candidate.box_xyxy, reference_box_xyxy)
        area_score = _area_similarity(candidate.box_xyxy, reference_box_xyxy)
        label_bonus = 0.10 if self.target is not None and candidate.label == self.target.label else -0.10
        return (
            candidate.score * 0.30
            + iou_score * 0.35
            + max(0.0, distance_score) * 0.20
            + area_score * 0.15
            + label_bonus
        )

    def _update_target_from_box(
        self,
        frame_rgb: np.ndarray,
        frame_index: int,
        box_xyxy: list[float],
        score: float,
        label: str,
        detection_source: str,
    ) -> None:
        if self.target is None:
            return
        should_refresh_segmentation = self._should_refresh_segmentation(frame_index, detection_source)
        smoothed_box = smooth_box_xyxy(
            self.target.smoothed_box_xyxy,
            clamp_box_xyxy(box_xyxy, frame_rgb.shape),
            alpha=self.config.box_smoothing_alpha,
        )
        self.target.raw_box_xyxy = clamp_box_xyxy(box_xyxy, frame_rgb.shape)
        self.target.smoothed_box_xyxy = clamp_box_xyxy(smoothed_box, frame_rgb.shape)
        self.target.score = float(score)
        self.target.label = label
        self.target.last_seen_frame = frame_index
        self.target.status = "locked"
        self.target.miss_count = 0
        self.target.tracking_score = float(score)
        if detection_source == "det":
            self.target.last_detection_frame = frame_index
        if should_refresh_segmentation:
            self._refresh_target_mask(frame_rgb, frame_index)

    def _apply_tracking_output(
        self,
        frame_rgb: np.ndarray,
        frame_index: int,
        tracking_output: dict,
        score: float,
        label: str,
        source: str,
    ) -> None:
        if self.target is None:
            return
        current_mask = tracking_output.get("mask")
        current_box = tracking_output.get("box_xyxy")
        if current_mask is None or current_box is None or int(current_mask.sum()) == 0:
            self._mark_target_lost()
            return
        self.target.mask_prob, self.target.mask = blend_binary_masks(
            self.target.mask_prob,
            current_mask,
            alpha=self.config.mask_smoothing_alpha,
        )
        mask_box = mask_to_box_xyxy(self.target.mask)
        reference_box = mask_box if mask_box is not None else current_box
        smoothed_box = smooth_box_xyxy(
            self.target.smoothed_box_xyxy,
            clamp_box_xyxy(reference_box, frame_rgb.shape),
            alpha=self.config.box_smoothing_alpha,
        )
        self.target.raw_box_xyxy = clamp_box_xyxy(current_box, frame_rgb.shape)
        self.target.smoothed_box_xyxy = clamp_box_xyxy(smoothed_box, frame_rgb.shape)
        self.target.score = float(score)
        self.target.tracking_score = float(tracking_output.get("score", score))
        self.target.label = label
        self.target.last_seen_frame = frame_index
        self.target.status = "locked"
        self.target.miss_count = 0
        if source == "det":
            self.target.last_detection_frame = frame_index

    def _mark_target_lost(self) -> None:
        if self.target is None:
            return
        self.target.miss_count += 1
        self.target.status = "lost"

    def _should_refresh_segmentation(self, frame_index: int, detection_source: str) -> bool:
        if self.target is None:
            return False
        if detection_source == "det":
            return True
        if self.config.segmentation_interval <= 1:
            return True
        return (frame_index - self.target.last_detection_frame) % self.config.segmentation_interval == 0

    def _refresh_target_mask(self, frame_rgb: np.ndarray, frame_index: int) -> None:
        if self.target is None or self.segmenter is None:
            return
        segmentation_box = expand_box_xyxy(
            self.target.smoothed_box_xyxy,
            self.config.segmentation_expand_ratio,
            frame_rgb.shape,
        )
        current_mask = self.segmenter.segment_box(frame_rgb, segmentation_box)
        if current_mask is None or current_mask.sum() == 0:
            current_mask = make_box_mask(frame_rgb.shape, self.target.smoothed_box_xyxy)
        self.target.mask_prob, self.target.mask = blend_binary_masks(
            self.target.mask_prob,
            current_mask,
            alpha=self.config.mask_smoothing_alpha,
        )
        mask_box = mask_to_box_xyxy(self.target.mask)
        if mask_box is not None:
            self.target.smoothed_box_xyxy = clamp_box_xyxy(
                smooth_box_xyxy(
                    self.target.smoothed_box_xyxy,
                    mask_box,
                    alpha=self.config.box_smoothing_alpha,
                ),
                frame_rgb.shape,
            )
        self.target.last_seen_frame = frame_index

    def _passes_geometry_filters(
        self,
        box_xyxy: list[float],
        image_shape: tuple[int, int] | tuple[int, int, int],
    ) -> bool:
        width, height = box_xyxy_size(box_xyxy)
        aspect_ratio = width / max(height, 1.0)
        if aspect_ratio < self.config.candidate_min_aspect_ratio:
            return False
        if aspect_ratio > self.config.candidate_max_aspect_ratio:
            return False
        image_area = float(image_shape[0] * image_shape[1])
        area_ratio = box_xyxy_area(box_xyxy) / max(image_area, 1.0)
        if area_ratio < self.config.min_box_area_ratio:
            return False
        if area_ratio > self.config.max_box_area_ratio:
            return False
        return True


def _box_center_in_box(inner_box_xyxy: list[float], outer_box_xyxy: list[float]) -> bool:
    center_x, center_y = box_xyxy_center(inner_box_xyxy)
    x1, y1, x2, y2 = outer_box_xyxy
    return x1 <= center_x <= x2 and y1 <= center_y <= y2


def _center_distance(box_xyxy: list[float], point_box_xyxy: list[float]) -> float:
    ax, ay = box_xyxy_center(box_xyxy)
    bx, by = box_xyxy_center(point_box_xyxy)
    return float(np.hypot(ax - bx, ay - by))


def _normalized_center_distance(box_a_xyxy: list[float], box_b_xyxy: list[float]) -> float:
    distance = _center_distance(box_a_xyxy, box_b_xyxy)
    aw, ah = box_xyxy_size(box_a_xyxy)
    bw, bh = box_xyxy_size(box_b_xyxy)
    norm = max(1.0, np.hypot((aw + bw) * 0.5, (ah + bh) * 0.5))
    return min(1.0, float(distance / norm))


def _area_similarity(box_a_xyxy: list[float], box_b_xyxy: list[float]) -> float:
    area_a = box_xyxy_area(box_a_xyxy)
    area_b = box_xyxy_area(box_b_xyxy)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    return float(min(area_a, area_b) / max(area_a, area_b))
