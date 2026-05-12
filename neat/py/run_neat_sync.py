"""
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are
 proprietary to SiMa and may be covered by U.S. and Foreign Patents,
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is
 strictly forbidden unless prior written permission is obtained from
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who
 have executed Confidentiality and Non-disclosure agreements explicitly
 covering such access.

 The copyright notice above does not evidence any actual or intended
 publication or disclosure  of  this source code, which includes information
 that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.

 ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
 DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
 LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
 REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.

**************************************************************************
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import (  # noqa: E402
    COCO_CLASSES,
    INPUT_H,
    INPUT_W,
    draw_detections,
    get_image_paths,
    prepare_output_dir,
    preprocess_image,
    postprocess_yolov8x_p2_4o,
)

try:
    import pyneat
except ImportError:
    sys.exit(
        "pyneat is not importable. Activate the NEAT Python environment first, "
        "or run this script through dk/devkit-run."
    )


DEFAULT_MODEL = (
    REPO_ROOT / "build/yolov8x-p2_opt_4o/yolov8x-p2_opt_4o_mpk.tar.gz"
)
DEFAULT_IMAGES = REPO_ROOT / "test_images"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run synchronous NEAT YOLOv8x-p2 inference on test images."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--conf-thr", type=float, default=0.25)
    parser.add_argument("--iou-thr", type=float, default=0.45)
    return parser.parse_args(argv[1:])


def read_bgr_image(path: Path) -> np.ndarray:
    """Read a BGR image from disk and fail with a useful error."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def build_model(model_path: Path, max_width: int, max_height: int):
    """Create a NEAT model configured for RGB image input and CVU preproc."""
    def set_preproc_option(name: str, value: object) -> None:
        """Set a preproc option only when this pyneat build exposes it."""
        if hasattr(opt.preproc, name):
            setattr(opt.preproc, name, value)

    opt = pyneat.ModelOptions()
    opt.media_type = "video/x-raw"
    opt.format = "RGB"
    opt.input_max_width = int(max_width)
    opt.input_max_height = int(max_height)
    opt.input_max_depth = 3

    set_preproc_option("input_width", int(max_width))
    set_preproc_option("input_height", int(max_height))
    set_preproc_option("output_width", INPUT_W)
    set_preproc_option("output_height", INPUT_H)
    set_preproc_option("normalize", True)
    set_preproc_option("aspect_ratio", True)
    set_preproc_option("q_scale", 254.9999849195601)
    set_preproc_option("q_zp", -128)
    set_preproc_option("input_img_type", "RGB")
    set_preproc_option("output_img_type", "RGB")
    set_preproc_option("output_dtype", "EVXX_INT8")
    set_preproc_option("padding_type", "CENTER")
    set_preproc_option("scaling_type", "BILINEAR")

    return pyneat.Model(str(model_path), opt)


def set_optional_input_limits(input_opt, max_width: int, max_height: int) -> None:
    """Set optional pyneat input caps limits when the installed API exposes them."""
    for attr, value in (
        ("max_width", max_width),
        ("max_height", max_height),
        ("max_depth", 3),
    ):
        if hasattr(input_opt, attr):
            setattr(input_opt, attr, int(value))


def build_sync_run(model, seed_rgb: np.ndarray, max_width: int, max_height: int):
    """Build an explicit synchronous preproc, MLA, detess-dequant NEAT run."""
    input_opt = model.input_appsrc_options(False)
    input_opt.media_type = "video/x-raw"
    input_opt.format = "RGB"
    input_opt.width = int(seed_rgb.shape[1])
    input_opt.height = int(seed_rgb.shape[0])
    input_opt.depth = 3
    set_optional_input_limits(input_opt, max_width, max_height)

    session = pyneat.Session()
    session.add(pyneat.nodes.input(input_opt))
    session.add(model.preprocess())
    session.add(pyneat.groups.mla(model))
    session.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)))
    session.add(pyneat.nodes.output())

    seed_tensor = pyneat.Tensor.from_numpy(
        seed_rgb,
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
    )
    run_options = pyneat.RunOptions()
    run_options.queue_depth = 1
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    run_options.output_memory = pyneat.OutputMemory.Owned
    return session.build(seed_tensor, pyneat.RunMode.Sync, run_options)


def iter_tensors(sample) -> Iterable:
    """Yield all tensor payloads from a NEAT sample or nested bundle."""
    if sample is None:
        return
    if sample.tensor is not None:
        yield sample.tensor
    for field in sample.fields or []:
        yield from iter_tensors(field)


def tensor_to_nhwc(tensor) -> np.ndarray:
    """Convert a NEAT tensor to a contiguous NHWC numpy array."""
    array = tensor.to_numpy(copy=True)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = np.expand_dims(array, axis=0)
    if array.ndim != 4:
        raise RuntimeError(f"expected rank-3 or rank-4 model output, got {array.shape}")
    return np.ascontiguousarray(array)


def collect_yolo_outputs(sample) -> list[np.ndarray]:
    """Collect YOLO head tensors in the order expected by the postprocessor."""
    outputs = [tensor_to_nhwc(tensor) for tensor in iter_tensors(sample)]
    if len(outputs) != 8:
        raise RuntimeError(f"expected 8 dequantized output tensors, got {len(outputs)}")

    bbox = sorted(
        [out for out in outputs if out.shape[-1] == 64],
        key=lambda out: out.shape[1],
        reverse=True,
    )
    classes = sorted(
        [out for out in outputs if out.shape[-1] == 80],
        key=lambda out: out.shape[1],
        reverse=True,
    )
    if len(bbox) != 4 or len(classes) != 4:
        shapes = ", ".join(str(out.shape) for out in outputs)
        raise RuntimeError(f"unexpected YOLO output tensor shapes: {shapes}")
    return bbox + classes


def run_one_image(run, image_path: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """Run inference for one image and write the overlay image."""
    image_bgr = read_bgr_image(image_path)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

    if not run.push(tensor):
        raise RuntimeError(f"push failed for {image_path}")

    sample = run.pull(timeout_ms=args.timeout_ms)
    if sample is None:
        raise TimeoutError(f"timed out waiting for output for {image_path}")

    outputs = collect_yolo_outputs(sample)
    boxes, scores, class_ids = postprocess_yolov8x_p2_4o(
        outputs,
        conf_thr=args.conf_thr,
        iou_thr=args.iou_thr,
        apply_class_sigmoid=True,
    )

    _, padded_bgr = preprocess_image(image_bgr, do_transpose=False)
    overlay = draw_detections(padded_bgr, boxes, scores, class_ids, COCO_CLASSES)
    output_path = output_dir / f"{image_path.stem}.jpg"
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"failed to write output image: {output_path}")
    return int(scores.shape[0])


def validate_inputs(args: argparse.Namespace) -> list[Path]:
    """Validate model and image paths, then return sorted image paths."""
    if not args.model.is_file():
        raise FileNotFoundError(f"model archive not found: {args.model}")
    image_paths = [Path(path) for path in get_image_paths(str(args.images))]
    if not image_paths:
        raise RuntimeError(f"no images found in: {args.images}")
    return image_paths


def main(argv: list[str]) -> int:
    """Program entry point."""
    args = parse_args(argv)
    image_paths = validate_inputs(args)
    prepare_output_dir(str(args.results))

    shapes = [read_bgr_image(path).shape for path in image_paths]
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)
    first_bgr = read_bgr_image(image_paths[0])
    first_rgb = np.ascontiguousarray(cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB))

    model = build_model(args.model, max_width=max_width, max_height=max_height)
    run = build_sync_run(model, first_rgb, max_width=max_width, max_height=max_height)

    try:
        for index, image_path in enumerate(image_paths, start=1):
            detections = run_one_image(run, image_path, args.results, args)
            print(
                f"[{index}/{len(image_paths)}] {image_path.name} -> "
                f"{args.results / (image_path.stem + '.jpg')} ({detections} detections)"
            )
    finally:
        run.close()

    print(f"wrote results to {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
