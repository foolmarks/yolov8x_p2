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
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils  # noqa: E402


DEFAULT_MODEL = REPO_ROOT / "build/yolov8x-p2_opt_4o/yolov8x-p2_opt_4o_mpk.tar.gz"
DEFAULT_IMAGES = REPO_ROOT / "test_images"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def load_pyneat() -> Any:
    """Import pyneat and provide a clear error when running off target."""
    try:
        import pyneat
    except ImportError as exc:
        raise RuntimeError(
            "pyneat is not importable. Run this on the Modalix DevKit with "
            "$HOME/pyneat/bin/python or activate $HOME/pyneat/bin/activate."
        ) from exc
    return pyneat


def get_rgb_input(img_bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to contiguous HWC RGB uint8 for NEAT appsrc."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(img_rgb)


def get_overlay_image(img_bgr: np.ndarray) -> np.ndarray:
    """Create the 640x640 padded BGR image used for drawing model-space boxes."""
    _, padded_bgr = utils.preprocess_image(img_bgr, do_transpose=False)
    return padded_bgr


def get_image_paths(image_dir: Path, limit: int | None) -> list[Path]:
    """Return sorted image paths from the configured image directory."""
    image_paths = [Path(path) for path in utils.get_image_paths(str(image_dir))]
    if limit is not None:
        return image_paths[:limit]
    return image_paths


def get_input_bounds(image_paths: list[Path]) -> tuple[int, int]:
    """Read image headers and return max width and height for dynamic appsrc caps."""
    max_width = 0
    max_height = 0
    for image_path in image_paths:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        height, width = img.shape[:2]
        max_width = max(max_width, width)
        max_height = max(max_height, height)

    if max_width <= 0 or max_height <= 0:
        raise RuntimeError("Could not read any images to determine input bounds")
    return max_width, max_height


def make_model_options(pyneat: Any, max_width: int, max_height: int) -> Any:
    """Configure the MPK generic CVU preproc for RGB input to 640x640 NHWC."""
    opts = pyneat.ModelOptions()
    opts.media_type = "video/x-raw"
    opts.format = "RGB"
    opts.input_max_width = max_width
    opts.input_max_height = max_height
    opts.input_max_depth = 3
    opts.original_width = utils.INPUT_W
    opts.original_height = utils.INPUT_H

    opts.preproc.input_img_type = "RGB"
    opts.preproc.output_img_type = "RGB"
    opts.preproc.output_width = utils.INPUT_W
    opts.preproc.output_height = utils.INPUT_H
    opts.preproc.scaled_width = utils.INPUT_W
    opts.preproc.scaled_height = utils.INPUT_H
    opts.preproc.normalize = True
    opts.preproc.aspect_ratio = True
    opts.preproc.padding_type = "CENTER"
    opts.preproc.scaling_type = "BILINEAR"
    return opts


def make_run_options(pyneat: Any, queue_depth: int) -> Any:
    """Create deterministic synchronous run options."""
    opts = pyneat.RunOptions()
    opts.queue_depth = max(queue_depth, 1)
    opts.overflow_policy = pyneat.OverflowPolicy.Block
    opts.output_memory = pyneat.OutputMemory.Auto
    return opts


def sample_to_outputs(sample: Any, pyneat: Any) -> list[np.ndarray]:
    """Convert a NEAT output sample or bundle into ordered NumPy tensors."""
    if sample is None:
        raise TimeoutError("Timed out waiting for NEAT pipeline output")

    if sample.kind == pyneat.SampleKind.Tensor:
        return [sample.tensor.to_numpy(copy=True)]

    if sample.kind != pyneat.SampleKind.Bundle:
        raise RuntimeError(f"Unsupported NEAT sample kind: {sample.kind}")

    fields = list(sample.fields)
    if len(fields) != 8:
        raise RuntimeError(f"Expected 8 output tensors, got {len(fields)}")

    def sort_key(field: Any) -> int:
        """Sort bundle fields by NEAT output index."""
        output_index = getattr(field, "output_index", -1)
        return output_index if output_index >= 0 else len(fields)

    outputs: list[np.ndarray] = []
    for field in sorted(fields, key=sort_key):
        if field.kind != pyneat.SampleKind.Tensor:
            raise RuntimeError(f"Bundle field is not a tensor: {field.kind}")
        output = field.tensor.to_numpy(copy=True)
        if output.ndim != 4 or output.shape[0] != 1:
            raise RuntimeError(f"Unexpected output tensor shape: {output.shape}")
        outputs.append(output)
    return outputs


def build_runner(pyneat: Any, model: Any, input_rgb: np.ndarray, queue_depth: int) -> Any:
    """Build a synchronous NEAT session for the current RGB input dimensions."""
    session = pyneat.Session()
    session.add(model.session())
    return session.build(
        input_rgb,
        mode=pyneat.RunMode.Sync,
        options=make_run_options(pyneat, queue_depth),
        copy=True,
        layout=pyneat.TensorLayout.HWC,
        image_format=pyneat.PixelFormat.RGB,
    )


def process_images(args: argparse.Namespace) -> None:
    """Run all images through NEAT, postprocess detections, and write overlays."""
    pyneat = load_pyneat()

    model_path = args.model.resolve()
    image_dir = args.images.resolve()
    output_dir = args.output.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Compiled model archive not found: {model_path}")

    image_paths = get_image_paths(image_dir, args.limit)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    utils.prepare_output_dir(str(output_dir))
    print(f"Found {len(image_paths)} image(s) in {image_dir}", flush=True)
    print(f"Results will be written to {output_dir}", flush=True)

    max_width, max_height = get_input_bounds(image_paths)
    model = pyneat.Model(
        str(model_path),
        make_model_options(pyneat, max_width, max_height),
    )

    for image_path in image_paths:
        print(f"Processing image: {image_path.name}", flush=True)
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  WARNING: Could not read image, skipping: {image_path}", flush=True)
            continue

        input_rgb = get_rgb_input(img_bgr)
        runner = build_runner(pyneat, model, input_rgb, args.queue_depth)
        try:
            pushed = runner.push(
                input_rgb,
                copy=True,
                layout=pyneat.TensorLayout.HWC,
                image_format=pyneat.PixelFormat.RGB,
            )
            if not pushed:
                raise RuntimeError(f"NEAT pipeline rejected input: {image_path}")

            outputs = sample_to_outputs(runner.pull(args.timeout_ms), pyneat)
            boxes, scores, class_ids = utils.postprocess_yolov8x_p2_4o(
                outputs,
                conf_thr=args.conf_thres,
                iou_thr=args.iou_thres,
                num_classes=80,
                apply_class_sigmoid=True,
            )

            if boxes.shape[0] == 0:
                print("  No detections above confidence threshold.", flush=True)
                annotated = get_overlay_image(img_bgr)
            else:
                print(f"  Detections: {boxes.shape[0]}", flush=True)
                annotated = utils.draw_detections(
                    get_overlay_image(img_bgr),
                    boxes,
                    scores,
                    class_ids,
                    utils.COCO_CLASSES,
                )

            output_path = output_dir / image_path.name
            if not cv2.imwrite(str(output_path), annotated):
                raise RuntimeError(f"Failed to write output image: {output_path}")
            print(f"  Annotated image written to: {output_path}", flush=True)
        finally:
            runner.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8x-p2 MPK through synchronous pyneat."
    )
    parser.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("-i", "--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("-l", "--limit", type=int, default=None)
    parser.add_argument("--queue-depth", type=int, default=4)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("-ct", "--conf-thres", type=float, default=0.50)
    parser.add_argument("-it", "--iou-thres", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    process_images(parse_args())


if __name__ == "__main__":
    main()
