from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.grounded_sam_core import (
    DEFAULT_GDINO_CHECKPOINT,
    DEFAULT_GDINO_CONFIG,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG_PATH,
    build_empty_result,
    choose_device,
    ensure_exists,
    infer_on_rgb_image,
    load_realtime_models,
    save_individual_masks,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GroundingDINO + SAM on one image.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--text", required=True, help='Caption prompt, e.g. "person . dog ."')
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "run"), help="Output directory.")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="GroundingDINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")
    parser.add_argument("--segmenter-backend", choices=("sam2", "auto_fast", "grabcut", "sam", "mobile_sam"), default="sam2")
    parser.add_argument("--sam-variant", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--min-box-area-ratio", type=float, default=0.0, help="Keep boxes larger than this fraction of image area.")
    parser.add_argument("--max-box-area-ratio", type=float, default=1.0, help="Keep boxes smaller than this fraction of image area.")
    parser.add_argument("--gdino-config", default=str(DEFAULT_GDINO_CONFIG))
    parser.add_argument("--gdino-checkpoint", default=str(DEFAULT_GDINO_CHECKPOINT))
    parser.add_argument("--sam-checkpoint", default=None, help="Optional custom SAM checkpoint path.")
    parser.add_argument("--mobile-sam-checkpoint", default=None, help="Optional custom MobileSAM checkpoint path.")
    parser.add_argument("--sam2-config", default=str(DEFAULT_SAM2_CONFIG_PATH), help="SAM2 config path.")
    parser.add_argument("--sam2-checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT), help="SAM2 checkpoint path.")
    parser.add_argument("--grabcut-iterations", type=int, default=1, help="GrabCut iterations when using grabcut backend.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = ensure_exists(Path(args.image), "Provide a valid input image path.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    image_rgb = np.asarray(Image.open(image_path).convert("RGB")).copy()
    result = infer_on_rgb_image(
        image_rgb=image_rgb,
        text_prompt=args.text,
        gdino_model=gdino_model,
        predictor=segmenter,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
        min_box_area_ratio=args.min_box_area_ratio,
        max_box_area_ratio=args.max_box_area_ratio,
    )

    if not result["detections"]:
        write_json(output_dir / "result.json", build_empty_result(str(image_path), args.text, device))
        Image.fromarray(image_rgb).save(output_dir / "detections.jpg")
        Image.fromarray(image_rgb).save(output_dir / "grounded_sam.jpg")
        return

    result["detections_image"].save(output_dir / "detections.jpg")
    result["result_image"].save(output_dir / "grounded_sam.jpg")
    mask_files = save_individual_masks(output_dir, result["masks"])

    payload = {
        "image": str(image_path),
        "text_prompt": args.text,
        "device": device,
        "segmenter_backend": segmenter.backend_name,
        "detections": [
            {**detection, "mask_file": mask_file}
            for detection, mask_file in zip(result["detections"], mask_files)
        ],
    }
    write_json(output_dir / "result.json", payload)


if __name__ == "__main__":
    main()
