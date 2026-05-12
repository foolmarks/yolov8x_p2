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
import time
from pathlib import Path

import cv2
import numpy as np

from run_neat_sync import (
    COCO_CLASSES,
    DEFAULT_IMAGES,
    DEFAULT_MODEL,
    DEFAULT_RESULTS,
    build_model,
    collect_yolo_outputs,
    draw_detections,
    get_image_paths,
    prepare_output_dir,
    preprocess_image,
    postprocess_yolov8x_p2_4o,
    pyneat,
    read_bgr_image,
    set_optional_input_limits,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run asynchronous NEAT YOLOv8x-p2 inference on test images."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--conf-thr", type=float, default=0.25)
    parser.add_argument("--iou-thr", type=float, default=0.45)
    parser.add_argument("--queue-depth", type=int, default=4)
    parser.add_argument("--max-in-flight", type=int, default=1)
    parser.add_argument("--poll-timeout-ms", type=int, default=100)
    return parser.parse_args(argv[1:])


def validate_inputs(args: argparse.Namespace) -> list[Path]:
    """Validate model and image paths, then return sorted image paths."""
    if not args.model.is_file():
        raise FileNotFoundError(f"model archive not found: {args.model}")
    image_paths = [Path(path) for path in get_image_paths(str(args.images))]
    if not image_paths:
        raise RuntimeError(f"no images found in: {args.images}")
    return image_paths


def build_async_run(
    model,
    seed_rgb: np.ndarray,
    max_width: int,
    max_height: int,
    queue_depth: int,
):
    """Build an explicit asynchronous preproc, MLA, detess-dequant NEAT run."""
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
    run_options.queue_depth = int(queue_depth)
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    run_options.output_memory = pyneat.OutputMemory.Owned
    return session.build(seed_tensor, pyneat.RunMode.Async, run_options)


def make_rgb_tensor(image_path: Path):
    """Read an image and convert it to an RGB NEAT tensor."""
    image_bgr = read_bgr_image(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    return pyneat.Tensor.from_numpy(
        image_rgb,
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
    )


def write_result(
    sample,
    image_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Postprocess one output sample and write its overlay image."""
    outputs = collect_yolo_outputs(sample)
    boxes, scores, class_ids = postprocess_yolov8x_p2_4o(
        outputs,
        conf_thr=args.conf_thr,
        iou_thr=args.iou_thr,
        apply_class_sigmoid=True,
    )

    image_bgr = read_bgr_image(image_path)
    _, padded_bgr = preprocess_image(image_bgr, do_transpose=False)
    overlay = draw_detections(padded_bgr, boxes, scores, class_ids, COCO_CLASSES)
    output_path = output_dir / f"{image_path.stem}.jpg"
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"failed to write output image: {output_path}")
    return int(scores.shape[0])


def try_admit_image(
    run,
    image_path: Path,
    in_flight: list[Path],
) -> bool:
    """Try to push one image without blocking and remember it if accepted."""
    tensor = make_rgb_tensor(image_path)
    accepted = run.try_push(tensor)
    if accepted:
        in_flight.append(image_path)
    return bool(accepted)


def run_async_loop(run, image_paths: list[Path], args: argparse.Namespace) -> int:
    """Run separate non-blocking push and pull operations until all images finish."""
    if not run.can_push():
        raise RuntimeError("built pipeline does not support pushed input")
    if not run.can_pull():
        raise RuntimeError("built pipeline does not support pulled output")
    if not run.running():
        raise RuntimeError("pipeline is not running after build")

    next_input = 0
    processed = 0
    in_flight: list[Path] = []
    max_in_flight = max(1, int(args.max_in_flight))
    deadline = time.monotonic() + max(1.0, args.timeout_ms / 1000.0)

    while processed < len(image_paths):
        made_progress = False

        while next_input < len(image_paths) and len(in_flight) < max_in_flight:
            if not run.running():
                raise RuntimeError(f"pipeline stopped before push: {run.last_error()}")
            if not try_admit_image(run, image_paths[next_input], in_flight):
                break
            next_input += 1
            made_progress = True

        sample = run.pull(timeout_ms=max(1, int(args.poll_timeout_ms)))
        if sample is not None:
            if not in_flight:
                raise RuntimeError("received output with no matching in-flight image")
            image_path = in_flight.pop(0)
            detections = write_result(sample, image_path, args.results, args)
            processed += 1
            made_progress = True
            print(
                f"[{processed}/{len(image_paths)}] {image_path.name} -> "
                f"{args.results / (image_path.stem + '.jpg')} ({detections} detections)"
            )

        if made_progress:
            deadline = time.monotonic() + max(1.0, args.timeout_ms / 1000.0)
            continue

        if time.monotonic() > deadline:
            raise TimeoutError(
                "timed out waiting for async pipeline progress; "
                f"processed={processed}, pushed={next_input}, in_flight={len(in_flight)}, "
                f"last_error={run.last_error()}"
            )

    return processed


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
    run = build_async_run(
        model,
        first_rgb,
        max_width=max_width,
        max_height=max_height,
        queue_depth=args.queue_depth,
    )

    try:
        processed = run_async_loop(run, image_paths, args)
        if processed != len(image_paths):
            raise RuntimeError(f"processed {processed} of {len(image_paths)} images")
    finally:
        run.close()

    print(f"wrote results to {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
