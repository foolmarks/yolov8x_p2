#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
 publication or disclosure  of  this
"""

import argparse
import os, sys
import shutil

from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort

import utils



def scale_boxes_to_original(boxes_640: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    """
    Scale boxes from letterboxed (INPUT_H, INPUT_W) space back to original image size.

    Must match the same letterboxing definition used in preprocess_image():
      - r = min(target_w / orig_w, target_h / orig_h)
      - new_w = round(orig_w * r), new_h = round(orig_h * r)
      - dw = target_w - new_w, dh = target_h - new_h
      - pad_left = dw / 2, pad_top = dh / 2
    """
    if boxes_640.size == 0:
        return boxes_640

    target_w, target_h = utils.INPUT_W, utils.INPUT_H

    # Recompute the same scale & padding used during preprocessing
    r = min(target_w / float(orig_w), target_h / float(orig_h))
    new_w = float(round(orig_w * r))
    new_h = float(round(orig_h * r))

    dw = target_w - new_w
    dh = target_h - new_h

    pad_left = dw / 2.0
    pad_top  = dh / 2.0

    boxes_orig = boxes_640.copy().astype(np.float32)

    # Undo padding, then undo scaling
    boxes_orig[:, [0, 2]] = (boxes_orig[:, [0, 2]] - pad_left) / r
    boxes_orig[:, [1, 3]] = (boxes_orig[:, [1, 3]] - pad_top)  / r

    # Clip to original image bounds
    boxes_orig[:, 0] = np.clip(boxes_orig[:, 0], 0, orig_w - 1)
    boxes_orig[:, 1] = np.clip(boxes_orig[:, 1], 0, orig_h - 1)
    boxes_orig[:, 2] = np.clip(boxes_orig[:, 2], 0, orig_w - 1)
    boxes_orig[:, 3] = np.clip(boxes_orig[:, 3], 0, orig_h - 1)

    return boxes_orig

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid.
    """
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax along the given axis.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def _decode_single_level(
    bbox_level: np.ndarray,
    cls_level: np.ndarray,
    stride: int,
    reg_max: int = 16,
    num_classes: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode one YOLOv8x-p2 head level.

    Args:
        bbox_level: (1, 64, H, W)  - 4 sides * reg_max bins (DFL logits).
        cls_level:  (1, 80, H, W)  - class probabilities (sigmoid already in graph).
        stride:     stride for this level (e.g. 4, 8, 16, 32).
        reg_max:    number of bins used in DFL (64 = 4*16 -> 16).
        num_classes: number of classes (80 for COCO).

    Returns:
        boxes_xyxy: (H*W, 4) in input-image pixels (model 640x640 space).
        scores:     (H*W, num_classes) class probabilities.
    """
    # Remove batch dimension
    bbox = bbox_level[0]  # (64, H, W)
    cls = cls_level[0]    # (80, H, W)

    _, H, W = bbox.shape

    # ---- Decode bbox with DFL ----
    # (64, H, W) -> (4, reg_max, H, W)
    bbox = bbox.reshape(4, reg_max, H, W)
    # -> (H, W, 4, reg_max)
    bbox = np.transpose(bbox, (2, 3, 0, 1))
    # -> (H*W, 4, reg_max)
    bbox = bbox.reshape(-1, 4, reg_max)

    # Softmax over reg_max dimension to get per-bin probabilities
    bbox_prob = _softmax(bbox, axis=-1)  # (N, 4, reg_max)

    # Expected value over bins [0, reg_max-1]
    bin_indices = np.arange(reg_max, dtype=np.float32)
    distances = bbox_prob @ bin_indices  # (N, 4)
    distances = distances * float(stride)

    # Build grid of centers in input-image pixels (640x640 space)
    ys = (np.arange(H, dtype=np.float32) + 0.5) * float(stride)
    xs = (np.arange(W, dtype=np.float32) + 0.5) * float(stride)
    xv, yv = np.meshgrid(xs, ys)  # (H, W)

    centers = np.stack((xv, yv), axis=-1).reshape(-1, 2)  # (N, 2)

    l = distances[:, 0]
    t = distances[:, 1]
    r = distances[:, 2]
    b = distances[:, 3]

    cx = centers[:, 0]
    cy = centers[:, 1]

    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b

    boxes_xyxy = np.stack((x1, y1, x2, y2), axis=-1)  # (N, 4)

    # ---- Class scores (already probabilities from graph) ----
    # (80, H, W) -> (80, H*W) -> (H*W, 80)
    cls = cls.reshape(num_classes, -1).transpose(1, 0)
    scores = cls  # no extra sigmoid

    return boxes_xyxy, scores


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between a single box and an array of boxes.

    Args:
        box:   (4,)   [x1, y1, x2, y2]
        boxes: (N, 4)

    Returns:
        ious: (N,)
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    inter = w * h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - inter
    iou = inter / np.maximum(union, 1e-7)
    return iou


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> np.ndarray:
    """
    Class-agnostic NMS.

    Args:
        boxes:   (N, 4)
        scores:  (N,)
        iou_thr: IoU threshold.

    Returns:
        keep_indices: (M,) indices of kept boxes.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = _box_iou(boxes[i], boxes[order[1:]])
        remaining = np.where(ious < iou_thr)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.int64)


def postprocess_yolov8x_p2_4o(
    outputs: list,
    conf_thr: float = 0.25,
    iou_thr: float = 0.45,
    num_classes: int = 80,
    apply_class_sigmoid: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Post-process for 4-output-pair YOLOv8x-p2 ONNX model.

    Args:
        outputs: list of 8 tensors in this exact order (ONNX output order):
                 [bbox_0, bbox_1, bbox_2, bbox_3,
                  class_prob_0, class_prob_1, class_prob_2, class_prob_3]
        conf_thr: confidence threshold on max class score.
        iou_thr:  NMS IoU threshold.
        num_classes: number of classes (80 for COCO).

    Returns:
        final_boxes:   (M, 4) xyxy in input-image pixels (model 640x640 space).
        final_scores:  (M,)   max class scores.
        final_classes: (M,)   integer class IDs.
    """
    assert len(outputs) == 8, "Expected 8 outputs (4 bbox + 4 class_prob)."

    # 160, 80, 40, 20 feature maps -> strides for 640x640 input
    strides = [4, 8, 16, 32]

    bbox_levels = outputs[0:4]
    cls_levels = outputs[4:8]

    all_boxes = []
    all_scores = []

    for level in range(4):
        bbox_level = bbox_levels[level]
        cls_level = cls_levels[level]
        stride = strides[level]

        boxes_xyxy, scores = _decode_single_level(
            bbox_level=bbox_level,
            cls_level=cls_level,
            stride=stride,
            reg_max=16,
            num_classes=num_classes,
        )
        all_boxes.append(boxes_xyxy)
        all_scores.append(scores)

    boxes = np.concatenate(all_boxes, axis=0)    # (N_total, 4) in 640x640 space
    scores = np.concatenate(all_scores, axis=0)  # (N_total, num_classes)

    if apply_class_sigmoid:
        scores = _sigmoid(scores)

    # Max class score per candidate
    class_ids = np.argmax(scores, axis=1)
    class_scores = scores[np.arange(scores.shape[0]), class_ids]

    # Confidence threshold
    mask = class_scores >= conf_thr
    boxes = boxes[mask]
    class_scores = class_scores[mask]
    class_ids = class_ids[mask]

    if boxes.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    # NMS (class-agnostic)
    keep = _nms(boxes, class_scores, iou_thr=iou_thr)

    final_boxes = boxes[keep]
    final_scores = class_scores[keep]
    final_classes = class_ids[keep]

    return final_boxes, final_scores, final_classes


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Letterbox to (INPUT_H, INPUT_W) while preserving aspect ratio.

    Definition of letterboxing used here:
      - r = min(target_w / w0, target_h / h0)
      - new_w = round(w0 * r), new_h = round(h0 * r)
      - dw = target_w - new_w, dh = target_h - new_h
      - pad_left   = dw // 2
      - pad_right  = dw - pad_left
      - pad_top    = dh // 2
      - pad_bottom = dh - pad_top
      - pad with black borders

    Then:
      - BGR -> RGB
      - normalize to [0,1]
      - HWC -> CHW
      - add batch dim -> (1,3,H,W)
    """
    h0, w0 = img_bgr.shape[:2]  # original height, width
    target_w, target_h = utils.INPUT_W, utils.INPUT_H

    # Scale to fit in target size while preserving aspect ratio
    r = min(target_w / w0, target_h / h0)
    new_w = int(round(w0 * r))
    new_h = int(round(h0 * r))

    # Resize with aspect ratio preserved
    if (w0, h0) != (new_w, new_h):
        img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_bgr.copy()

    # Compute padding
    dw = target_w - new_w  # total padding width
    dh = target_h - new_h  # total padding height

    pad_left   = dw // 2
    pad_right  = dw - pad_left
    pad_top    = dh // 2
    pad_bottom = dh - pad_top

    # Pad with black borders (0,0,0)
    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # BGR -> RGB, normalize, HWC -> CHW, add batch dim
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # (1,3,H,W)

    return img




def draw_detections(img_bgr, boxes, scores, class_ids, class_names=None):
    if boxes.size == 0:
        return img_bgr

    if class_names is None:
        class_names = [str(i) for i in range(int(class_ids.max()) + 1)]

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        cls_id = int(cls_id)
        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
        caption = f"{label} {score:.2f}"

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img_bgr,
            (x1, y1 - th - baseline),
            (x1 + tw, y1),
            (0, 255, 0),
            thickness=-1,
        )
        cv2.putText(
            img_bgr,
            caption,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return img_bgr



def get_image_paths(folder: str):
    """
    Return a list of full paths to image files in 'folder'.
    Image files are detected by extension.
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {folder}")

    files = sorted(os.listdir(folder))
    image_paths = [
        os.path.join(folder, f)
        for f in files
        if os.path.splitext(f.lower())[1] in valid_exts
    ]
    return image_paths



def prepare_output_dir(output_dir: str):
    """
    If output_dir exists, delete it and its contents, then recreate it.
    If it doesn't exist, create it.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def implement(args) -> None:

    # Prepare output folder
    prepare_output_dir(args.output_dir)

    # Load ONNX model
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"ONNX model not found: {args.model}")

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name


    # Get all image paths from input folder
    image_paths = get_image_paths(args.input_dir)
    if len(image_paths) == 0:
        print(f"No image files found in folder: {args.input_dir}")
        return

    print(f"Found {len(image_paths)} image(s) in '{args.input_dir}'")
    print(f"Output images will be written to '{args.output_dir}'")


    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"\nProcessing image: {filename}", flush=True)

        # Load original image (any size)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  WARNING: Could not read image, skipping: {img_path}")
            continue

        orig_h, orig_w = img_bgr.shape[:2]

        # Preprocess (resize → tensor)
        img_input = preprocess_image(img_bgr)

        print("Running inference...", flush=True)
        # Outputs are returned in the order defined by the ONNX graph:
        # [bbox_0, bbox_1, bbox_2, bbox_3,
        #  class_prob_0, class_prob_1, class_prob_2, class_prob_3]
        outputs = session.run(None, {input_name: img_input})

        # Postprocess in 640x640 space
        boxes_640, scores, class_ids = postprocess_yolov8x_p2_4o(
            outputs,
            conf_thr=args.conf_thres,
            iou_thr=args.iou_thres,
            num_classes=80,
            apply_class_sigmoid=True
            )


        if boxes_640.shape[0] == 0:
            print("  No detections above confidence threshold.")
            annotated = img_bgr.copy()
        else:
            print(f"  Detections: {boxes_640.shape[0]}")

            # Scale boxes back to original image size
            boxes_orig = scale_boxes_to_original(boxes_640, orig_w, orig_h)   
            annotated = draw_detections(img_bgr.copy(), boxes_orig, scores, class_ids, utils.COCO_CLASSES)

        out_path = os.path.join(args.output_dir, filename)
        ok = cv2.imwrite(out_path, annotated)
        if not ok:
            raise RuntimeError(f"Failed to write output image: {out_path}")
        print(f"  Annotated image written to: {out_path}")

    return

def run_main():
  
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(description="Run YOLOv8x-p2 ONNX model with 4 bbox + 4 class_prob outputs.")
    ap.add_argument("--input-dir",   type=str,   default="./test_images",           help="Path to input image folder")
    ap.add_argument( "--model",      type=str,   default="yolov8x-p2_opt_4o.onnx",  help="Path to the ONNX model file.")
    ap.add_argument("--output-dir",  type=str,   default="./build/onnx_4_pred",     help="Path to output folder for annotated images")
    ap.add_argument("--conf-thres",  type=float, default=0.45,                      help="Confidence threshold")
    ap.add_argument("--iou-thres",   type=float, default=0.45,                      help="IoU threshold for NMS")
    args = ap.parse_args()

    print('\n' + utils.DIVIDER, flush=True)
    print(sys.version, flush=True)
    print(utils.DIVIDER, flush=True)

    implement(args)

    return



if __name__ == "__main__":
    run_main()
