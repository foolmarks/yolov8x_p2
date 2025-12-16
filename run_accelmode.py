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

"""
Author: Mark Harvey
"""


import argparse
import os
import sys
from pathlib import Path

import cv2

import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Palette-specific imports
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.apis.model import Model
from afe.apis.release_v1 import get_model_sdk_version
from afe.core.utils import length_hinted

DIVIDER = "-" * 50


def implement(args):
    enable_verbose_error_messages()

    """
    Make results folder
    """
    build_dir = Path(args.build_dir).resolve()
    annotated_images = (build_dir / "accel_pred").resolve()

    # If the folder exists, delete it first
    utils.prepare_output_dir(f"{annotated_images}")
    print(f"Annotated images will be written to {annotated_images}", flush=True)

    """
    load quantized model
    """
    model_path = f"{args.build_dir}/{args.model_name}"
    print(f"Loading {args.model_name} quantized model from {model_path}", flush=True)
    quant_model = Model.load(f"{args.model_name}.sima", model_path)

    """
    Prepare test data
      - create list of dictionaries
      - Each dictionary key is an input name, value is a preprocessed data sample
    """
    image_paths = utils.get_image_paths(args.test_dir)
    num_test_images = min(args.num_test_images, len(image_paths))
    print(f"Using {num_test_images} out of {len(image_paths)}  test images", flush=True)
    image_paths = image_paths[:num_test_images]

    test_data = []
    original_images = []
    # original_dims = []
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing image: {filename}", flush=True)

        # Load original image (any size)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  WARNING: Could not read image, skipping: {img_path}")
            continue

        orig_h, orig_w = img_bgr.shape[:2]

        inputs = dict()
        data, bgr_padded = utils.preprocess_image(img_bgr, do_transpose=False)
        inputs["images"] = data
        original_images.append(bgr_padded)
        # original_dims.append((orig_h, orig_w))
        test_data.append(inputs)

    """
    Run in accel mode
    Returns a list of lists of np arrays
    Outer list length = num_test_images
    Inner list lengths = number of model outputs = 8
    Np array shapes:
            (1, 160, 160, 64)
            (1, 80, 80, 64)
            (1, 40, 40, 64)
            (1, 20, 20, 64)
            (1, 160, 160, 80)
            (1, 80, 80, 80)
            (1, 40, 40, 80)
            (1, 20, 20, 80)
    """
    pred = quant_model.execute_in_accelerator_mode(
        input_data=length_hinted(num_test_images, test_data),
        devkit=args.hostname,
        username=args.username,
        password=args.password,
    )

    print("Model is executed in accelerator mode.", flush=True)

    """
    Evaluate results
    """
    for i, p in enumerate(pred):
        # Postprocess in 640x640 space
        boxes_640, scores, class_ids = utils.postprocess_yolov8x_p2_4o(
            p,
            conf_thr=args.conf_thres,
            iou_thr=args.iou_thres,
            num_classes=80,
            apply_class_sigmoid=True,
        )

        if boxes_640.shape[0] == 0:
            print("  No detections above confidence threshold.")
            annotated = original_images[i]
        else:
            print(f"  Detections: {boxes_640.shape[0]}")

        # Scale boxes back to original image size
        # orig_w = original_dims[i][1]
        # orig_h = original_dims[i][0]
        # boxes_orig = utils.scale_boxes_to_original(boxes_640, orig_w, orig_h)
        # annotated = utils.draw_detections(
        #    original_images[i], boxes_orig, scores, class_ids, utils.COCO_CLASSES
        # )
        annotated = utils.draw_detections(
            original_images[i], boxes_640, scores, class_ids, utils.COCO_CLASSES
        )

        filename = os.path.basename(image_paths[i])
        out_path = os.path.join(annotated_images, filename)
        ok = cv2.imwrite(out_path, annotated)
        if not ok:
            raise RuntimeError(f"Failed to write output image: {out_path}")
        print(f"  Annotated image written to: {out_path}")

    return


def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-bd",
        "--build_dir",
        type=str,
        default="build",
        help="Path of build folder. Default is build",
    )
    ap.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="yolov8x-p2_opt_4o",
        help="quantized model name",
    )
    ap.add_argument(
        "-td",
        "--test_dir",
        type=str,
        default="./test_images",
        help="Path to test images folder. Default is ./test_images",
    )
    ap.add_argument(
        "-ti",
        "--num_test_images",
        type=int,
        default=10,
        help="Number of test images. Default is 10",
    )
    ap.add_argument(
        "-u",
        "--username",
        type=str,
        default="sima",
        help="Target device user name. Default is sima",
    )
    ap.add_argument(
        "-p",
        "--password",
        type=str,
        default="edgeai",
        help="Target device password. Default is edgeai",
    )
    ap.add_argument(
        "-hn",
        "--hostname",
        type=str,
        default="192.168.1.21",
        help="Target device IP address. Default is 192.168.1.21",
    )
    ap.add_argument(
        "-ct", "--conf_thres", type=float, default=0.45, help="Confidence threshold"
    )
    ap.add_argument(
        "-it", "--iou_thres", type=float, default=0.45, help="IoU threshold for NMS"
    )
    args = ap.parse_args()

    print("\n" + DIVIDER, flush=True)
    print("Model SDK version", get_model_sdk_version())
    print(sys.version, flush=True)
    print(DIVIDER, flush=True)

    implement(args)


if __name__ == "__main__":
    run_main()
