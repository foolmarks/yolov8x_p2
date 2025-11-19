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
import cv2
import onnxruntime as ort

import utils




def implement(args) -> None:

    # Prepare output folder
    utils.prepare_output_dir(args.output_dir)

    # Load ONNX model
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"ONNX model not found: {args.model}")

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name


    # Get all image paths from input folder
    image_paths = utils.get_image_paths(args.input_dir)
    if len(image_paths) == 0:
        print(f"No image files found in folder: {args.input_dir}")
        return

    print(f"Found {len(image_paths)} image(s) in '{args.input_dir}'")
    print(f"Output images will be written to '{args.output_dir}'")


    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing image: {filename}", flush=True)

        # Load original image (any size)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  WARNING: Could not read image, skipping: {img_path}")
            continue

        orig_h, orig_w = img_bgr.shape[:2]

        # Preprocess (resize → tensor)
        img_input = utils.preprocess_image(img_bgr)

        print("Running inference...", flush=True)
        # Outputs are returned in the order defined by the ONNX graph:
        # [bbox_0, bbox_1, bbox_2, bbox_3,
        #  class_prob_0, class_prob_1, class_prob_2, class_prob_3]
        outputs = session.run(None, {input_name: img_input})


        # Postprocess in 640x640 space
        boxes_640, scores, class_ids = utils.postprocess_yolov8x_p2_4o(
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
            boxes_orig = utils.scale_boxes_to_original(boxes_640, orig_w, orig_h)   
            annotated = utils.draw_detections(img_bgr.copy(), boxes_orig, scores, class_ids, utils.COCO_CLASSES)

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
    ap.add_argument("--output_dir",  type=str,   default="./build/onnx_4_pred",     help="Path to output folder for annotated images")
    ap.add_argument("--conf_thres",  type=float, default=0.45,                      help="Confidence threshold")
    ap.add_argument("--iou_thres",   type=float, default=0.45,                      help="IoU threshold for NMS")
    args = ap.parse_args()

    print('\n' + utils.DIVIDER, flush=True)
    print(sys.version, flush=True)
    print(utils.DIVIDER, flush=True)

    implement(args)

    return



if __name__ == "__main__":
    run_main()
