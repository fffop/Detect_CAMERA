from __future__ import annotations

import json
import os
import sys
import ctypes
import importlib
import warnings
from collections import OrderedDict
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
GROUNDINGDINO_REPO = ROOT / "external" / "GroundingDINO"
SAM_REPO = ROOT / "external" / "segment-anything"
MOBILE_SAM_REPO = ROOT / "external" / "MobileSAM"
SAM2_REPO = ROOT / "external" / "SAM2"
DEFAULT_TEXT_ENCODER_DIR = ROOT / "checkpoints" / "bert-base-uncased"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")


def _configure_warning_filters() -> None:
    warning_rules = [
        ("ignore", r".*torch\.utils\._pytree\._register_pytree_node.*", FutureWarning),
        ("ignore", r".*Importing from timm\.models\.layers is deprecated.*", FutureWarning),
        ("ignore", r".*torch\.meshgrid: in an upcoming release.*", UserWarning),
        ("ignore", r".*You are using `torch\.load` with `weights_only=False`.*", FutureWarning),
        ("ignore", r".*The `device` argument is deprecated and will be removed in v5 of Transformers.*", FutureWarning),
        ("ignore", r".*torch\.utils\.checkpoint: the use_reentrant parameter should be passed explicitly.*", UserWarning),
        ("ignore", r".*None of the inputs have requires_grad=True.*", UserWarning),
    ]
    for action, message, category in warning_rules:
        warnings.filterwarnings(action, message=message, category=category)


_configure_warning_filters()


def _prepare_cuda_runtime_paths() -> None:
    conda_prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = conda_prefix / "lib" / py_ver / "site-packages"
    torch_lib = site_packages / "torch" / "lib"
    nvidia_root = site_packages / "nvidia"
    candidate_dirs = [torch_lib]
    if nvidia_root.exists():
        candidate_dirs.extend(sorted(path for path in nvidia_root.glob("*/lib") if path.is_dir()))
    candidate_dirs.extend(
        [
            Path("/usr/local/cuda/lib64"),
            Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        ]
    )

    existing_dirs = [str(path) for path in candidate_dirs if path.exists()]
    if not existing_dirs:
        return

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in current_ld.split(":") if part]
    for lib_dir in reversed(existing_dirs):
        if lib_dir not in ld_parts:
            ld_parts.insert(0, lib_dir)
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    for lib_name in (
        "libc10.so",
        "libtorch_cpu.so",
        "libtorch_python.so",
        "libc10_cuda.so",
        "libtorch_cuda.so",
        "libnvrtc.so.12",
        "libnvrtc.so.13",
        "libcudart.so.12",
        "libcudart.so.13",
    ):
        for lib_dir in existing_dirs:
            lib_path = Path(lib_dir) / lib_name
            if lib_path.exists():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


_prepare_cuda_runtime_paths()

import torch
import transformers

for repo_path in (GROUNDINGDINO_REPO, SAM_REPO, MOBILE_SAM_REPO, SAM2_REPO):
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

from groundingdino.models import build_model  # noqa: E402
import groundingdino.datasets.transforms as gdino_transforms  # noqa: E402
from groundingdino.util import box_ops  # noqa: E402
from groundingdino.util.misc import clean_state_dict  # noqa: E402
from groundingdino.util.slconfig import SLConfig  # noqa: E402
from groundingdino.util.utils import get_phrases_from_posmap  # noqa: E402
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402

DEFAULT_GDINO_CONFIG = (
    ROOT / "external" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
)
DEFAULT_GDINO_CHECKPOINT = ROOT / "checkpoints" / "groundingdino_swint_ogc.pth"
DEFAULT_SAM_CHECKPOINTS = {
    "vit_b": ROOT / "checkpoints" / "sam_vit_b_01ec64.pth",
    "vit_l": ROOT / "checkpoints" / "sam_vit_l_0b3195.pth",
    "vit_h": ROOT / "checkpoints" / "sam_vit_h_4b8939.pth",
}
DEFAULT_MOBILE_SAM_CHECKPOINT = ROOT / "checkpoints" / "mobile_sam.pt"
DEFAULT_SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_SAM2_CONFIG_PATH = SAM2_REPO / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml"
DEFAULT_SAM2_CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
MASK_COLORS = [
    (255, 99, 71),
    (30, 144, 255),
    (34, 139, 34),
    (255, 165, 0),
    (148, 0, 211),
    (255, 20, 147),
]

_GDINO_PREPROCESS = gdino_transforms.Compose(
    [
        gdino_transforms.RandomResize([800], max_size=1333),
        gdino_transforms.ToTensor(),
        gdino_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def box_xyxy_iou(box_a: list[float] | tuple[float, float, float, float], box_b: list[float] | tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def deduplicate_box_candidates(
    boxes_xyxy: torch.Tensor,
    logits: torch.Tensor,
    phrases: list[str],
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if boxes_xyxy.numel() == 0 or iou_threshold <= 0:
        return boxes_xyxy, logits, phrases

    sorted_indices = sorted(
        range(len(phrases)),
        key=lambda idx: float(logits[idx]),
        reverse=True,
    )
    kept_indices: list[int] = []
    kept_boxes: list[list[float]] = []

    for idx in sorted_indices:
        candidate_box = boxes_xyxy[idx].tolist()
        if any(box_xyxy_iou(candidate_box, kept_box) >= iou_threshold for kept_box in kept_boxes):
            continue
        kept_indices.append(idx)
        kept_boxes.append(candidate_box)

    keep_tensor = torch.tensor(kept_indices, dtype=torch.long, device=boxes_xyxy.device)
    return boxes_xyxy[keep_tensor], logits[keep_tensor], [phrases[idx] for idx in kept_indices]


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def configure_torch_runtime(device: str) -> None:
    if device != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True


def ensure_compatible_transformers() -> None:
    major_version = int(transformers.__version__.split(".", 1)[0])
    if major_version >= 5:
        raise RuntimeError(
            "GroundingDINO is incompatible with transformers>=5 in this setup. "
            f"Found transformers=={transformers.__version__}. "
            "Please run: conda run -n SAM_DINO python -m pip install 'transformers>=4.30,<5'"
        )


def ensure_exists(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}\nHint: {hint}")
    return path


def resolve_model_paths(
    gdino_config: str,
    gdino_checkpoint: str,
    sam_variant: str,
    sam_checkpoint: str | None,
) -> tuple[Path, Path, Path]:
    resolved_gdino_config = ensure_exists(
        Path(gdino_config),
        "Run `bash scripts/setup_project.sh SAM_DINO cpu` to clone the official GroundingDINO repo.",
    )
    resolved_gdino_checkpoint = ensure_exists(
        Path(gdino_checkpoint),
        "Run `bash scripts/download_weights.sh vit_b` to download model weights.",
    )
    resolved_sam_checkpoint = ensure_exists(
        Path(sam_checkpoint) if sam_checkpoint else DEFAULT_SAM_CHECKPOINTS[sam_variant],
        "Run `bash scripts/download_weights.sh vit_b` or choose another SAM checkpoint.",
    )
    return resolved_gdino_config, resolved_gdino_checkpoint, resolved_sam_checkpoint


def resolve_text_encoder_source() -> str:
    configured = os.environ.get("GROUNDINGDINO_TEXT_ENCODER")
    if configured:
        return configured
    if DEFAULT_TEXT_ENCODER_DIR.exists():
        return str(DEFAULT_TEXT_ENCODER_DIR)
    return "bert-base-uncased"


def load_gdino_model(
    model_config_path: str,
    model_checkpoint_path: str,
    device: str,
):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.text_encoder_type = resolve_text_encoder_source()
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.to(device)
    model.eval()
    return model


def load_models(
    device: str,
    gdino_config: str,
    gdino_checkpoint: str,
    sam_variant: str,
    sam_checkpoint: str | None,
):
    ensure_compatible_transformers()
    configure_torch_runtime(device)
    resolved_gdino_config, resolved_gdino_checkpoint, resolved_sam_checkpoint = resolve_model_paths(
        gdino_config=gdino_config,
        gdino_checkpoint=gdino_checkpoint,
        sam_variant=sam_variant,
        sam_checkpoint=sam_checkpoint,
    )
    gdino_model = load_gdino_model(
        model_config_path=str(resolved_gdino_config),
        model_checkpoint_path=str(resolved_gdino_checkpoint),
        device=device,
    )
    sam_model = sam_model_registry[sam_variant](checkpoint=str(resolved_sam_checkpoint))
    sam_model.to(device=device)
    predictor = SamPredictor(sam_model)
    return gdino_model, predictor


class BaseBoxSegmenter:
    backend_name = "none"

    def segment_box(
        self,
        image_rgb: np.ndarray,
        box_xyxy: list[float] | tuple[float, float, float, float],
    ) -> np.ndarray:
        raise NotImplementedError


class SamBoxSegmenter(BaseBoxSegmenter):
    backend_name = "sam"

    def __init__(self, predictor: SamPredictor, device: str, backend_name: str = "sam"):
        self.predictor = predictor
        self.device = device
        self.backend_name = backend_name

    def segment_box(
        self,
        image_rgb: np.ndarray,
        box_xyxy: list[float] | tuple[float, float, float, float],
    ) -> np.ndarray:
        clamped_box = clamp_box_xyxy(box_xyxy, image_rgb.shape)
        box_tensor = torch.tensor([clamped_box], dtype=torch.float32)
        self.predictor.set_image(image_rgb)
        transformed_box = self.predictor.transform.apply_boxes_torch(box_tensor, image_rgb.shape[:2]).to(self.device)
        with torch.inference_mode():
            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_box,
                multimask_output=False,
            )
        return masks[0, 0].detach().cpu().numpy().astype(np.uint8)


class Sam2BoxSegmenter(BaseBoxSegmenter):
    backend_name = "sam2"

    def __init__(self, predictor, device: str):
        self.predictor = predictor
        self.device = device

    def segment_box(
        self,
        image_rgb: np.ndarray,
        box_xyxy: list[float] | tuple[float, float, float, float],
    ) -> np.ndarray:
        clamped_box = np.asarray(clamp_box_xyxy(box_xyxy, image_rgb.shape), dtype=np.float32)
        self.predictor.set_image(image_rgb)
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=clamped_box,
                multimask_output=True,
                return_logits=False,
                normalize_coords=False,
            )
        best_mask = None
        best_quality = float("-inf")
        for mask, score in zip(masks, scores):
            processed_mask = select_prompt_component(mask.astype(np.uint8), clamped_box)
            quality = score_prompt_mask(processed_mask, clamped_box, float(score))
            if quality > best_quality:
                best_quality = quality
                best_mask = processed_mask
        if best_mask is None:
            return make_box_mask(image_rgb.shape, clamped_box)
        return best_mask


class Sam2VideoRealtimeTracker:
    backend_name = "sam2_video"

    def __init__(
        self,
        predictor,
        device: str,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
        max_frames_per_track: int = 96,
    ):
        self.predictor = predictor
        self.device = torch.device(device)
        self.offload_video_to_cpu = bool(offload_video_to_cpu)
        self.offload_state_to_cpu = bool(offload_state_to_cpu)
        self.frame_storage_device = torch.device("cpu") if self.offload_video_to_cpu else self.device
        self.storage_device = torch.device("cpu") if self.offload_state_to_cpu else self.device
        self.max_frames_per_track = max(0, int(max_frames_per_track))
        self._img_mean = torch.tensor(
            (0.485, 0.456, 0.406),
            dtype=torch.float32,
            device=self.frame_storage_device,
        )[:, None, None]
        self._img_std = torch.tensor(
            (0.229, 0.224, 0.225),
            dtype=torch.float32,
            device=self.frame_storage_device,
        )[:, None, None]
        self.reset()

    def reset(self) -> None:
        self.inference_state: dict | None = None
        self.obj_id = 1
        self._next_frame_idx = 0
        self._last_box_xyxy: list[float] | None = None

    def initialize(
        self,
        frame_rgb: np.ndarray,
        box_xyxy: list[float] | tuple[float, float, float, float],
    ) -> dict | None:
        if self.inference_state is not None:
            self.inference_state = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        frame_tensor = self._frame_to_tensor(frame_rgb)
        self.inference_state = self._build_inference_state(
            first_frame_tensor=frame_tensor,
            video_height=frame_rgb.shape[0],
            video_width=frame_rgb.shape[1],
        )
        self._next_frame_idx = 1
        self._last_box_xyxy = [float(v) for v in box_xyxy]
        with self._autocast_context():
            _, _, video_res_masks = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=self.obj_id,
                box=np.asarray(box_xyxy, dtype=np.float32),
                points=None,
                labels=None,
            )
            self.predictor.propagate_in_video_preflight(self.inference_state)
        return self._build_tracking_output(video_res_masks[0], score=1.0)

    def track(self, frame_rgb: np.ndarray) -> dict | None:
        if self.inference_state is None:
            return None
        if self.max_frames_per_track > 0 and self._next_frame_idx >= self.max_frames_per_track:
            if self._last_box_xyxy is None:
                return None
            return self.initialize(frame_rgb, self._last_box_xyxy)
        frame_tensor = self._frame_to_tensor(frame_rgb)
        frame_idx = self._next_frame_idx
        self.inference_state["images"].append(frame_tensor)
        self.inference_state["num_frames"] = len(self.inference_state["images"])
        obj_idx = self.predictor._obj_id_to_idx(self.inference_state, self.obj_id)
        obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        with self._autocast_context():
            current_out, pred_masks_gpu = self.predictor._run_single_frame_inference(
                inference_state=self.inference_state,
                output_dict=obj_output_dict,
                frame_idx=frame_idx,
                batch_size=1,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
        obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out
        self.inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": False}
        with self._autocast_context():
            _, video_res_masks = self.predictor._get_orig_video_res_output(self.inference_state, pred_masks_gpu)
        self._next_frame_idx += 1
        score_logits = current_out["object_score_logits"].detach().float().reshape(-1)
        score = float(score_logits.sigmoid()[0].item()) if score_logits.numel() > 0 else 1.0
        return self._build_tracking_output(video_res_masks[0], score=score)

    def _frame_to_tensor(self, frame_rgb: np.ndarray) -> torch.Tensor:
        image_size = int(self.predictor.image_size)
        resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        tensor = (
            torch.from_numpy(resized)
            .permute(2, 0, 1)
            .to(device=self.frame_storage_device, dtype=torch.float32)
            / 255.0
        )
        tensor = (tensor - self._img_mean) / self._img_std
        return tensor.contiguous()

    def _build_inference_state(
        self,
        first_frame_tensor: torch.Tensor,
        video_height: int,
        video_width: int,
    ) -> dict:
        inference_state = {
            "images": [first_frame_tensor],
            "num_frames": 1,
            "offload_video_to_cpu": self.offload_video_to_cpu,
            "offload_state_to_cpu": self.offload_state_to_cpu,
            "video_height": int(video_height),
            "video_width": int(video_width),
            "device": self.device,
            "storage_device": self.storage_device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        with self._autocast_context():
            self.predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _build_tracking_output(self, mask_logits: torch.Tensor, score: float) -> dict | None:
        mask_tensor = mask_logits.detach().float()
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor[0]
        mask = (mask_tensor > 0.0).to(torch.uint8).cpu().numpy()
        box_xyxy = mask_to_box_xyxy(mask)
        if box_xyxy is None or int(mask.sum()) == 0:
            return None
        self._last_box_xyxy = [float(v) for v in box_xyxy]
        return {
            "mask": mask.astype(np.uint8),
            "box_xyxy": box_xyxy,
            "score": float(score),
        }


class GrabCutBoxSegmenter(BaseBoxSegmenter):
    backend_name = "grabcut"

    def __init__(self, iterations: int = 1, expand_ratio: float = 1.15):
        self.iterations = max(1, int(iterations))
        self.expand_ratio = max(1.0, float(expand_ratio))

    def segment_box(
        self,
        image_rgb: np.ndarray,
        box_xyxy: list[float] | tuple[float, float, float, float],
    ) -> np.ndarray:
        expanded_box = expand_box_xyxy(box_xyxy, self.expand_ratio, image_rgb.shape)
        x1, y1, x2, y2 = box_xyxy_to_int_tuple(expanded_box)
        roi = image_rgb[y1 : y2 + 1, x1 : x2 + 1]
        if roi.size == 0:
            return make_box_mask(image_rgb.shape, box_xyxy)

        roi_height, roi_width = roi.shape[:2]
        if roi_width < 8 or roi_height < 8:
            return make_box_mask(image_rgb.shape, box_xyxy)

        inner_margin_x = max(1, int(round(roi_width * 0.08)))
        inner_margin_y = max(1, int(round(roi_height * 0.08)))
        rect_width = max(1, roi_width - 2 * inner_margin_x)
        rect_height = max(1, roi_height - 2 * inner_margin_y)
        rect = (inner_margin_x, inner_margin_y, rect_width, rect_height)

        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(
                cv2.cvtColor(roi, cv2.COLOR_RGB2BGR),
                mask,
                rect,
                bgd_model,
                fgd_model,
                self.iterations,
                cv2.GC_INIT_WITH_RECT,
            )
            foreground = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                1,
                0,
            ).astype(np.uint8)
        except cv2.error:
            foreground = make_box_mask(roi.shape, [0, 0, roi_width - 1, roi_height - 1])

        full_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        full_mask[y1 : y2 + 1, x1 : x2 + 1] = foreground[: y2 - y1 + 1, : x2 - x1 + 1]
        if full_mask.sum() == 0:
            return make_box_mask(image_rgb.shape, box_xyxy)
        return full_mask


def load_box_segmenter(
    device: str,
    backend: str,
    sam_variant: str = "vit_b",
    sam_checkpoint: str | None = None,
    mobile_sam_checkpoint: str | None = None,
    sam2_config: str | None = None,
    sam2_checkpoint: str | None = None,
    grabcut_iterations: int = 1,
) -> BaseBoxSegmenter:
    normalized_backend = backend.lower().strip()
    if normalized_backend == "grabcut":
        return GrabCutBoxSegmenter(iterations=grabcut_iterations)

    if normalized_backend in {"sam", "mobile_sam"}:
        return _load_sam_family_segmenter(
            device=device,
            backend=normalized_backend,
            sam_variant=sam_variant,
            sam_checkpoint=sam_checkpoint,
            mobile_sam_checkpoint=mobile_sam_checkpoint,
        )

    if normalized_backend == "sam2":
        return _load_sam2_segmenter(
            device=device,
            sam2_config=sam2_config,
            sam2_checkpoint=sam2_checkpoint,
        )

    if normalized_backend == "auto_fast":
        try:
            return _load_sam2_segmenter(
                device=device,
                sam2_config=sam2_config,
                sam2_checkpoint=sam2_checkpoint,
            )
        except (ModuleNotFoundError, FileNotFoundError, AttributeError, RuntimeError):
            pass
        try:
            return _load_sam_family_segmenter(
                device=device,
                backend="mobile_sam",
                sam_variant=sam_variant,
                sam_checkpoint=sam_checkpoint,
                mobile_sam_checkpoint=mobile_sam_checkpoint,
            )
        except (ModuleNotFoundError, FileNotFoundError, AttributeError, RuntimeError):
            return GrabCutBoxSegmenter(iterations=grabcut_iterations)

    raise ValueError(f"Unsupported segmenter backend: {backend}")


def load_realtime_models(
    device: str,
    gdino_config: str,
    gdino_checkpoint: str,
    segmenter_backend: str,
    sam_variant: str,
    sam_checkpoint: str | None,
    mobile_sam_checkpoint: str | None = None,
    sam2_config: str | None = None,
    sam2_checkpoint: str | None = None,
    grabcut_iterations: int = 1,
):
    ensure_compatible_transformers()
    configure_torch_runtime(device)
    resolved_gdino_config = ensure_exists(
        Path(gdino_config),
        "Run `bash scripts/setup_project.sh SAM_DINO cpu` to clone the official GroundingDINO repo.",
    )
    resolved_gdino_checkpoint = ensure_exists(
        Path(gdino_checkpoint),
        "Run `bash scripts/download_weights.sh vit_b` to download model weights.",
    )
    gdino_model = load_gdino_model(
        model_config_path=str(resolved_gdino_config),
        model_checkpoint_path=str(resolved_gdino_checkpoint),
        device=device,
    )
    segmenter = load_box_segmenter(
        device=device,
        backend=segmenter_backend,
        sam_variant=sam_variant,
        sam_checkpoint=sam_checkpoint,
        mobile_sam_checkpoint=mobile_sam_checkpoint,
        sam2_config=sam2_config,
        sam2_checkpoint=sam2_checkpoint,
        grabcut_iterations=grabcut_iterations,
    )
    return gdino_model, segmenter


def load_realtime_detector(
    device: str,
    gdino_config: str,
    gdino_checkpoint: str,
):
    ensure_compatible_transformers()
    configure_torch_runtime(device)
    resolved_gdino_config = ensure_exists(
        Path(gdino_config),
        "Run `bash scripts/setup_project.sh SAM_DINO cpu` to clone the official GroundingDINO repo.",
    )
    resolved_gdino_checkpoint = ensure_exists(
        Path(gdino_checkpoint),
        "Run `bash scripts/download_weights.sh vit_b` to download model weights.",
    )
    return load_gdino_model(
        model_config_path=str(resolved_gdino_config),
        model_checkpoint_path=str(resolved_gdino_checkpoint),
        device=device,
    )


def _load_sam_family_segmenter(
    device: str,
    backend: str,
    sam_variant: str,
    sam_checkpoint: str | None,
    mobile_sam_checkpoint: str | None,
) -> SamBoxSegmenter:
    if backend == "sam":
        resolved_checkpoint = ensure_exists(
            Path(sam_checkpoint) if sam_checkpoint else DEFAULT_SAM_CHECKPOINTS[sam_variant],
            "Run `bash scripts/download_weights.sh vit_b` or choose another SAM checkpoint.",
        )
        sam_model = sam_model_registry[sam_variant](checkpoint=str(resolved_checkpoint))
        sam_model.to(device=device)
        return SamBoxSegmenter(
            predictor=SamPredictor(sam_model),
            device=device,
            backend_name="sam",
        )

    try:
        mobile_sam = importlib.import_module("mobile_sam")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MobileSAM is not installed. Run `bash scripts/setup_project.sh SAM_DINO cpu` "
            "after the script has cloned MobileSAM, or install the package in the active environment."
        ) from exc
    predictor_cls = getattr(mobile_sam, "SamPredictor")
    registry = getattr(mobile_sam, "sam_model_registry")
    resolved_checkpoint = ensure_exists(
        Path(mobile_sam_checkpoint) if mobile_sam_checkpoint else DEFAULT_MOBILE_SAM_CHECKPOINT,
        "Provide a valid MobileSAM checkpoint, e.g. checkpoints/mobile_sam.pt.",
    )
    mobile_model = registry["vit_t"](checkpoint=str(resolved_checkpoint))
    mobile_model.to(device=device)
    return SamBoxSegmenter(
        predictor=predictor_cls(mobile_model),
        device=device,
        backend_name="mobile_sam",
    )


def _load_sam2_segmenter(
    device: str,
    sam2_config: str | None,
    sam2_checkpoint: str | None,
) -> Sam2BoxSegmenter:
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SAM2 is not installed. Install external/SAM2 into the active environment first."
        ) from exc

    resolved_config = ensure_exists(
        Path(sam2_config) if sam2_config else DEFAULT_SAM2_CONFIG_PATH,
        "Provide a valid SAM2 config path, e.g. external/SAM2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml.",
    )
    resolved_checkpoint = ensure_exists(
        Path(sam2_checkpoint) if sam2_checkpoint else DEFAULT_SAM2_CHECKPOINT,
        "Provide a valid SAM2 checkpoint, e.g. checkpoints/sam2.1_hiera_tiny.pt.",
    )

    config_name = _resolve_sam2_config_name(resolved_config, sam2_config)

    sam2_model = build_sam2(
        config_file=config_name,
        ckpt_path=str(resolved_checkpoint),
        device=device,
        mode="eval",
    )
    predictor = SAM2ImagePredictor(sam2_model)
    return Sam2BoxSegmenter(predictor=predictor, device=device)


def load_sam2_video_tracker(
    device: str,
    sam2_config: str | None = None,
    sam2_checkpoint: str | None = None,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    max_frames_per_track: int = 96,
):
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SAM2 video predictor is not installed. Install external/SAM2 into the active environment first."
        ) from exc

    resolved_config = ensure_exists(
        Path(sam2_config) if sam2_config else DEFAULT_SAM2_CONFIG_PATH,
        "Provide a valid SAM2 config path, e.g. external/SAM2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml.",
    )
    resolved_checkpoint = ensure_exists(
        Path(sam2_checkpoint) if sam2_checkpoint else DEFAULT_SAM2_CHECKPOINT,
        "Provide a valid SAM2 checkpoint, e.g. checkpoints/sam2.1_hiera_tiny.pt.",
    )
    config_name = _resolve_sam2_config_name(resolved_config, sam2_config)
    predictor = build_sam2_video_predictor(
        config_file=config_name,
        ckpt_path=str(resolved_checkpoint),
        device=device,
        mode="eval",
    )
    if device == "cuda":
        predictor = predictor.to(dtype=torch.bfloat16)
    return Sam2VideoRealtimeTracker(
        predictor=predictor,
        device=device,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        max_frames_per_track=max_frames_per_track,
    )


def _resolve_sam2_config_name(resolved_config: Path, raw_config: str | None) -> str:
    config_name = DEFAULT_SAM2_CONFIG_NAME
    if raw_config:
        try:
            config_name = str(resolved_config.resolve().relative_to((SAM2_REPO / "sam2").resolve()))
        except ValueError:
            config_parts = resolved_config.as_posix().split("/sam2/")
            if len(config_parts) == 2:
                config_name = config_parts[1]
    return config_name


def draw_boxes(image_rgb: np.ndarray, boxes_xyxy: torch.Tensor, labels: list[str]) -> Image.Image:
    image = Image.fromarray(image_rgb)
    drawer = ImageDraw.Draw(image)
    for idx, (box, label) in enumerate(zip(boxes_xyxy.tolist(), labels)):
        color = MASK_COLORS[idx % len(MASK_COLORS)]
        x1, y1, x2, y2 = [int(v) for v in box]
        drawer.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_y = max(0, y1 - 18)
        drawer.text((x1 + 4, text_y), label, fill=color)
    return image


def overlay_masks(image_rgb: np.ndarray, masks: torch.Tensor, labels: list[str], boxes_xyxy: torch.Tensor) -> Image.Image:
    base = image_rgb.astype(np.float32).copy()
    for idx, mask_tensor in enumerate(masks):
        mask = mask_tensor[0].detach().cpu().numpy().astype(bool)
        color = np.array(MASK_COLORS[idx % len(MASK_COLORS)], dtype=np.float32)
        base[mask] = 0.45 * base[mask] + 0.55 * color

    result = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    drawer = ImageDraw.Draw(result)
    for idx, (box, label) in enumerate(zip(boxes_xyxy.tolist(), labels)):
        color = MASK_COLORS[idx % len(MASK_COLORS)]
        x1, y1, x2, y2 = [int(v) for v in box]
        drawer.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_y = max(0, y1 - 18)
        drawer.text((x1 + 4, text_y), label, fill=color)
    return result


def save_individual_masks(output_dir: Path, masks: torch.Tensor, prefix: str = "mask") -> list[str]:
    filenames: list[str] = []
    for idx, mask_tensor in enumerate(masks):
        mask = (mask_tensor[0].detach().cpu().numpy().astype(np.uint8)) * 255
        mask_name = f"{prefix}_{idx:02d}.png"
        Image.fromarray(mask).save(output_dir / mask_name)
        filenames.append(mask_name)
    return filenames


def image_to_bgr_array(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def clamp_box_xyxy(
    box_xyxy: list[float] | tuple[float, float, float, float],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> list[float]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = min(max(x1, 0.0), max(0, width - 1))
    y1 = min(max(y1, 0.0), max(0, height - 1))
    x2 = min(max(x2, x1 + 1.0), max(1, width) - 1)
    y2 = min(max(y2, y1 + 1.0), max(1, height) - 1)
    return [x1, y1, x2, y2]


def box_xyxy_area(box_xyxy: list[float] | tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_xyxy_center(box_xyxy: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def box_xyxy_size(box_xyxy: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def box_xyxy_to_int_tuple(
    box_xyxy: list[float] | tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    return x1, y1, x2, y2


def expand_box_xyxy(
    box_xyxy: list[float] | tuple[float, float, float, float],
    expand_ratio: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> list[float]:
    cx, cy = box_xyxy_center(box_xyxy)
    width, height = box_xyxy_size(box_xyxy)
    scaled_width = width * max(1.0, expand_ratio)
    scaled_height = height * max(1.0, expand_ratio)
    expanded = [
        cx - scaled_width * 0.5,
        cy - scaled_height * 0.5,
        cx + scaled_width * 0.5,
        cy + scaled_height * 0.5,
    ]
    return clamp_box_xyxy(expanded, image_shape)


def normalized_roi_to_xyxy(
    roi: tuple[float, float, float, float] | None,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> list[float] | None:
    if roi is None:
        return None
    height, width = image_shape[:2]
    x1, y1, x2, y2 = roi
    absolute = [x1 * width, y1 * height, x2 * width, y2 * height]
    return clamp_box_xyxy(absolute, image_shape)


def smooth_box_xyxy(
    previous_box_xyxy: list[float] | tuple[float, float, float, float] | None,
    current_box_xyxy: list[float] | tuple[float, float, float, float],
    alpha: float,
) -> list[float]:
    current = [float(v) for v in current_box_xyxy]
    if previous_box_xyxy is None:
        return current
    alpha = min(max(alpha, 0.0), 1.0)
    previous = [float(v) for v in previous_box_xyxy]
    return [
        alpha * prev + (1.0 - alpha) * curr
        for prev, curr in zip(previous, current)
    ]


def blend_binary_masks(
    previous_mask_prob: np.ndarray | None,
    current_mask: np.ndarray | None,
    alpha: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if current_mask is None:
        if previous_mask_prob is None:
            return None, None
        return previous_mask_prob, (previous_mask_prob >= 0.5).astype(np.uint8)
    current_float = current_mask.astype(np.float32)
    if previous_mask_prob is None or previous_mask_prob.shape != current_float.shape:
        mask_prob = current_float
    else:
        alpha = min(max(alpha, 0.0), 1.0)
        mask_prob = alpha * previous_mask_prob + (1.0 - alpha) * current_float
    return mask_prob, (mask_prob >= 0.5).astype(np.uint8)


def mask_to_box_xyxy(mask: np.ndarray | None) -> list[float] | None:
    if mask is None:
        return None
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def select_prompt_component(
    mask: np.ndarray,
    prompt_box_xyxy: list[float] | tuple[float, float, float, float],
) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy_to_int_tuple(prompt_box_xyxy)
    constrained_mask = np.zeros_like(mask, dtype=np.uint8)
    prompt_region = mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)
    if prompt_region.size == 0 or prompt_region.sum() == 0:
        return constrained_mask

    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(prompt_region, connectivity=8)
    if component_count <= 1:
        constrained_mask[y1 : y2 + 1, x1 : x2 + 1] = prompt_region
        return constrained_mask

    center_x = int(round((x2 - x1) * 0.5))
    center_y = int(round((y2 - y1) * 0.5))
    best_component = 0
    best_score = float("-inf")
    prompt_area = max(1.0, float(prompt_region.shape[0] * prompt_region.shape[1]))

    for component_index in range(1, component_count):
        area = float(component_stats[component_index, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component_mask = component_labels == component_index
        component_height, component_width = component_mask.shape
        contains_center = (
            0 <= center_y < component_height
            and 0 <= center_x < component_width
            and bool(component_mask[center_y, center_x])
        )
        area_ratio = area / prompt_area
        score = area_ratio
        if contains_center:
            score += 1.0
        if area_ratio > 0.92:
            score -= 0.75
        if area_ratio < 0.02:
            score -= 0.25
        if score > best_score:
            best_score = score
            best_component = component_index

    if best_component <= 0:
        constrained_mask[y1 : y2 + 1, x1 : x2 + 1] = prompt_region
        return constrained_mask

    filtered_region = (component_labels == best_component).astype(np.uint8)
    constrained_mask[y1 : y2 + 1, x1 : x2 + 1] = filtered_region
    return constrained_mask


def score_prompt_mask(
    mask: np.ndarray,
    prompt_box_xyxy: list[float] | tuple[float, float, float, float],
    sam_score: float,
) -> float:
    x1, y1, x2, y2 = box_xyxy_to_int_tuple(prompt_box_xyxy)
    prompt_area = max(1.0, float((x2 - x1 + 1) * (y2 - y1 + 1)))
    prompt_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
    area = float(prompt_mask.sum())
    if area <= 0:
        return float("-inf")

    area_ratio = area / prompt_area
    center_x = min(prompt_mask.shape[1] - 1, max(0, int(round((x2 - x1) * 0.5))))
    center_y = min(prompt_mask.shape[0] - 1, max(0, int(round((y2 - y1) * 0.5))))
    center_bonus = 0.35 if prompt_mask[center_y, center_x] > 0 else 0.0

    target_ratio = 0.45
    shape_score = max(0.0, 1.0 - abs(area_ratio - target_ratio) / target_ratio)
    if area_ratio > 0.92:
        shape_score -= 0.75
    if area_ratio < 0.02:
        shape_score -= 0.35
    return shape_score + center_bonus + 0.15 * sam_score


def make_box_mask(
    image_shape: tuple[int, int] | tuple[int, int, int],
    box_xyxy: list[float] | tuple[float, float, float, float],
) -> np.ndarray:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box_xyxy_to_int_tuple(clamp_box_xyxy(box_xyxy, image_shape))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    return mask


def build_empty_result(source_name: str, text_prompt: str, device: str) -> dict:
    return {
        "source": source_name,
        "text_prompt": text_prompt,
        "device": device,
        "detections": [],
        "message": "No detections matched the prompt and thresholds.",
    }


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def resize_image_keep_aspect(image_rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    if max_side <= 0:
        return image_rgb, 1.0
    height, width = image_rgb.shape[:2]
    current_max_side = max(height, width)
    if current_max_side <= max_side:
        return image_rgb, 1.0
    scale = float(max_side) / float(current_max_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_image_for_gdino(image_bgr: np.ndarray) -> torch.Tensor:
    image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    image_transformed, _ = _GDINO_PREPROCESS(image_pillow, None)
    return image_transformed


def predict_gdino_fast(
    model,
    image_tensor: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    caption = preprocess_caption(caption)
    image_tensor = image_tensor.to(device, non_blocking=True)
    autocast_context = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if device == "cuda"
        else nullcontext()
    )

    with torch.inference_mode():
        with autocast_context:
            outputs = model(image_tensor[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].float().cpu().sigmoid()[0]
    prediction_boxes = outputs["pred_boxes"].float().cpu()[0]

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]
    boxes = prediction_boxes[mask]

    tokenizer = model.tokenizer
    cache = getattr(model, "_codex_caption_cache", None)
    if cache is None or cache.get("caption") != caption:
        tokenized = tokenizer(caption)
        model._codex_caption_cache = {"caption": caption, "tokenized": tokenized}
    else:
        tokenized = cache["tokenized"]
    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace(".", "")
        for logit in logits
    ]
    return boxes, logits.max(dim=1)[0], phrases


def infer_on_rgb_image(
    image_rgb: np.ndarray,
    text_prompt: str,
    gdino_model,
    predictor: BaseBoxSegmenter | SamPredictor | None,
    box_threshold: float,
    text_threshold: float,
    device: str,
    min_box_area_ratio: float = 0.0,
    max_box_area_ratio: float = 1.0,
    dedupe_iou_threshold: float = 0.65,
    apply_sam: bool = True,
    inference_max_side: int = 0,
) -> dict:
    original_height, original_width = image_rgb.shape[:2]
    detector_image_rgb, resize_scale = resize_image_keep_aspect(image_rgb, inference_max_side)
    image_bgr = cv2.cvtColor(detector_image_rgb, cv2.COLOR_RGB2BGR)
    processed_image = preprocess_image_for_gdino(image_bgr=image_bgr).to(device)
    boxes, logits, phrases = predict_gdino_fast(
        model=gdino_model,
        image_tensor=processed_image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    if boxes.numel() == 0:
        return {
            "boxes_xyxy": None,
            "logits": [],
            "phrases": [],
            "labels": [],
            "masks": None,
            "detections_image": Image.fromarray(image_rgb),
            "result_image": Image.fromarray(image_rgb),
            "detections": [],
        }

    det_height, det_width = detector_image_rgb.shape[:2]
    scale = torch.tensor([det_width, det_height, det_width, det_height], dtype=boxes.dtype, device=boxes.device)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * scale
    boxes_xyxy_cpu = boxes_xyxy.cpu()
    if resize_scale != 1.0:
        boxes_xyxy_cpu[:, [0, 2]] /= resize_scale
        boxes_xyxy_cpu[:, [1, 3]] /= resize_scale
    boxes_xyxy_cpu[:, [0, 2]] = boxes_xyxy_cpu[:, [0, 2]].clamp(0, original_width - 1)
    boxes_xyxy_cpu[:, [1, 3]] = boxes_xyxy_cpu[:, [1, 3]].clamp(0, original_height - 1)

    image_area = float(original_height * original_width)
    area_mask: list[bool] = []
    for box in boxes_xyxy_cpu.tolist():
        x1, y1, x2, y2 = box
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_ratio = box_area / image_area if image_area > 0 else 0.0
        area_mask.append(min_box_area_ratio <= area_ratio <= max_box_area_ratio)

    if not any(area_mask):
        return {
            "boxes_xyxy": None,
            "logits": [],
            "phrases": [],
            "labels": [],
            "masks": None,
            "detections_image": Image.fromarray(image_rgb),
            "result_image": Image.fromarray(image_rgb),
            "detections": [],
        }

    keep_indices = [idx for idx, keep in enumerate(area_mask) if keep]
    keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=boxes_xyxy_cpu.device)
    boxes_xyxy_cpu = boxes_xyxy_cpu[keep_tensor]
    logits = logits[keep_tensor.to(logits.device)]
    phrases = [phrases[idx] for idx in keep_indices]
    boxes_xyxy_cpu, logits, phrases = deduplicate_box_candidates(
        boxes_xyxy=boxes_xyxy_cpu,
        logits=logits,
        phrases=phrases,
        iou_threshold=dedupe_iou_threshold,
    )

    labels = [f"{phrase} ({float(logit):.3f})" for phrase, logit in zip(phrases, logits)]
    detections = [
        {
            "label": phrase,
            "score": round(float(logit), 6),
            "box_xyxy": [round(float(v), 2) for v in box],
        }
        for phrase, logit, box in zip(phrases, logits, boxes_xyxy_cpu.tolist())
    ]
    detections_image = draw_boxes(image_rgb, boxes_xyxy_cpu, labels)
    masks = None
    result_image = detections_image
    if apply_sam:
        if predictor is None:
            raise ValueError("predictor must be provided when apply_sam=True")
        if isinstance(predictor, BaseBoxSegmenter):
            mask_stack = [
                predictor.segment_box(image_rgb, box_xyxy).astype(np.uint8)
                for box_xyxy in boxes_xyxy_cpu.tolist()
            ]
            if mask_stack:
                masks = torch.from_numpy(np.stack(mask_stack, axis=0)[:, None, :, :])
        else:
            predictor.set_image(image_rgb)
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy_cpu, image_rgb.shape[:2]).to(device)
            with torch.inference_mode():
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
        if masks is not None:
            result_image = overlay_masks(image_rgb, masks, labels, boxes_xyxy_cpu)

    return {
        "boxes_xyxy": boxes_xyxy_cpu,
        "logits": logits,
        "phrases": phrases,
        "labels": labels,
        "masks": masks,
        "detections_image": detections_image,
        "result_image": result_image,
        "detections": detections,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
