'''
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
'''


'''
Run ONNX model on all images in a folder.
'''

import argparse
import os, sys
import shutil

import cv2
import numpy as np
import onnxruntime as ort

import utils

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



def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """
    Non-Maximum Suppression.
    boxes: (N,4) in x1,y1,x2,y2
    scores: (N,)
    Returns indices of boxes to keep.
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / (union + 1e-6)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def postprocess(outputs, conf_thres: float, iou_thres: float):
    """
    Decode YOLOv8 ONNX outputs and run NMS in 640x640 space.

    outputs: model output, expected shape (1, C, N) or (1, N, C)
             where C = 4 + num_classes, N = number of candidate boxes.
    Returns:
      boxes_xyxy (M,4), scores (M,), class_ids (M,)
      coords are in 640x640 image space.
    """
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]

    preds = np.squeeze(outputs, axis=0)  # (C,N) or (N,C)

    if preds.ndim != 2:
        raise ValueError(f"Unexpected predictions shape: {preds.shape}")

    # We want (N, C): one row per candidate
    if preds.shape[0] < preds.shape[1]:
        preds = preds.transpose(1, 0)

    boxes = preds[:, :4]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    # Confidence filter
    mask = scores >= conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if boxes.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32), np.array([]), np.array([])

    # YOLOv8 boxes: (cx, cy, w, h) in 640x640 space
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Clip to [0, 640)
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, utils.INPUT_W - 1)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, utils.INPUT_H - 1)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, utils.INPUT_W - 1)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, utils.INPUT_H - 1)

    # NMS
    keep = nms(boxes_xyxy, scores, iou_thres)

    return boxes_xyxy[keep], scores[keep], class_ids[keep]


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

        # Inference
        outputs = session.run(None, {input_name: img_input})

        # Postprocess in 640x640 space
        boxes_640, scores, class_ids = postprocess(
            outputs,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir",   type=str,   default="./test_images",      help="Path to input image folder")
    ap.add_argument("--model",       type=str,   default="yolov8x-p2.onnx",    help="Path to ONNX model")
    ap.add_argument("--output-dir",  type=str,   default="./build/onnx_pred",  help="Path to output folder for annotated images")
    ap.add_argument("--conf-thres",  type=float, default=0.45,                 help="Confidence threshold")
    ap.add_argument("--iou-thres",   type=float, default=0.45,                 help="IoU threshold for NMS")
    args = ap.parse_args()

    print('\n' + utils.DIVIDER, flush=True)
    print(sys.version, flush=True)
    print(utils.DIVIDER, flush=True)

    implement(args)

    return


if __name__ == "__main__":
    run_main()
