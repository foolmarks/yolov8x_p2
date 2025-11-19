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
Quantize, evaluate and compile model
'''


'''
Author: SiMa Technologies 
'''


import onnx
import sys, os
import argparse
import numpy as np
import tarfile 
import logging
import cv2
from pathlib import Path
from typing import List, Dict, Tuple

import utils


# Palette-specific imports
from afe.load.importers.general_importer import ImporterParams, onnx_source
from afe.apis.defines import default_quantization, QuantizationScheme, gen1_target, gen2_target, gen1_target, gen2_target, CalibrationMethod
from afe.ir.tensor_type import ScalarType
from afe.apis.loaded_net import load_model
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.apis.release_v1 import get_model_sdk_version
from afe.core.utils import length_hinted


DIVIDER = '-'*50


def _get_onnx_input_shapes_dtypes(model_path: str) -> Tuple[Dict, Dict]:
    """
    Load an ONNX model and return two dictionaries describing its *true* inputs,
    ignoring any graph initializers (weights/biases).

    Returns:
        shapes_by_input:
            { input_name: (d0, d1, ...) } where each dimension (dn) is:
              - int for fixed sizes,
              - str for symbolic dimensions (e.g., "batch", "N"),
              - None if the dimension is present but unknown,
              - or the entire value can be None if the tensor is rank-unknown.
        dtypes_by_input:
            { input_name: dtype } where:
              - if the ONNX dtype is float32 -> the value is the symbol ScalarType.float32
              - otherwise -> the original NumPy-style dtype string (e.g., 'float16', 'int64')
              - or None if it could not be determined.
    """
    # Parse and sanity-check the model graph structure.
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    # Filter out parameters that appear as graph inputs.
    initializer_names = {init.name for init in model.graph.initializer}

    # Plain dictionaries
    shapes_by_input = {}
    dtypes_by_input = {}

    # Iterate over declared graph inputs
    for vi in model.graph.input:
        if vi.name in initializer_names:
            continue  # not a real runtime input

        # Only handle tensor inputs
        if not vi.type.HasField("tensor_type"):
            continue

        ttype = vi.type.tensor_type

        # ----- dtype -----
        elem_type = ttype.elem_type
        np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(elem_type, None)

        if np_dtype is None:
            dtypes_by_input[vi.name] = None
        else:
            dtype_name = np_dtype.name  # e.g., 'float32', 'int64'
            if dtype_name == 'float32':
                dtypes_by_input[vi.name] = ScalarType.float32
            else:
                dtypes_by_input[vi.name] = dtype_name
                print(f'Warning - input {vi.name} is not float32')

        # ----- shape -----
        if not ttype.HasField("shape"):
            shapes_by_input[vi.name] = None  # rank-unknown
            continue

        dims_list = []
        for d in ttype.shape.dim:
            if d.HasField("dim_value"):
                dims_list.append(int(d.dim_value))       # fixed dimension
            elif d.HasField("dim_param"):
                dims_list.append(d.dim_param)            # symbolic dimension
            else:
                dims_list.append(None)                   # unknown dimension

        # Store as immutable tuple
        shapes_by_input[vi.name] = tuple(dims_list)

    return shapes_by_input, dtypes_by_input





def _data_prep(folder_path: str, num_images: int, input_shapes_dict: Dict) -> List[Dict]:
  '''
  Prepare data
  '''
  processed_data =[]
  image_paths = utils.get_image_paths(folder_path)
  num_images = min(num_images, len(image_paths))
  image_paths = image_paths[:num_images]

  # make a list of dictionaries
  # key = input name, value = pre-processed data
  for input_name in input_shapes_dict.keys():
    inputs = dict()
    for img_path in image_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  WARNING: Could not read image, skipping: {img_path}")
            continue

    inputs[input_name] = utils.preprocess_image(img_bgr,do_transpose=False)
    processed_data.append(inputs)

  return processed_data





def implement(args):

  # enable verbose error messages.
  enable_verbose_error_messages()


  '''
  Make results folder
  '''
  # Derive directory name from model filename (without extension)
  output_model_name = Path(args.model_path).stem

  build_dir = Path(args.build_dir).resolve()
  results_dir = (build_dir / output_model_name).resolve()

  # If the folder exists, delete it first
  utils.prepare_output_dir(f'{results_dir}')
  print(f"Results will be written to {results_dir}", flush=True)


  
  '''
  Load the floating-point ONNX model into Sima format
  input types & shapes are dictionaries
  input types dictionary: each key,value pair is an input name (string) and a type
  input shapes dictionary: each key,value pair is an input name (string) and a shape (tuple)
  '''
  input_shapes_dict, input_types_dict = _get_onnx_input_shapes_dtypes(args.model_path)
  print(DIVIDER)
  print('Model Inputs:')
  for name, dims in input_shapes_dict.items():
     print(f'{name}: {dims}')
  print(DIVIDER)
     
  # importer parameters
  importer_params: ImporterParams = onnx_source(model_path=args.model_path,
                                                shape_dict=input_shapes_dict,
                                                dtype_dict=input_types_dict)
  

  # select Gen 1 or Gen 2 as target device
  target = gen2_target if args.generation == 2 else gen1_target

  # load ONNX floating-point model into SiMa's LoadedNet format
  loaded_net = load_model(importer_params,target=target,log_level=logging.INFO)
  print(f'Loaded model from {args.model_path}',flush=True)



  '''
  Prepare calibration data
    - create list of dictionaries
    - Each dictionary key is an input name, value is a preprocessed data sample
  '''
  calib_data = _data_prep(args.calib_dir, args.num_calib_images, input_shapes_dict)


  '''
  Quantize
  '''
  print(f'Quantizing with {len(calib_data)} calibration samples',flush=True)



  # set number of quantization bits (INT8 or BF16)
  if (args.bf16):
      weights_quant_scheme=QuantizationScheme(asymmetric=False, per_channel=False, bf16=True)
      activ_quant_scheme=QuantizationScheme(asymmetric=False, per_channel=False, bf16=True)
  else:
      weights_quant_scheme=QuantizationScheme(asymmetric=False, per_channel=True, bits=8)
      activ_quant_scheme=QuantizationScheme(asymmetric=True, per_channel=False, bits=args.quant_bits)

  # set other quantization parameters
  quant_config = default_quantization.with_activation_quantization(activ_quant_scheme) \
                                     .with_weight_quantization(weights_quant_scheme) \
                                     .with_bias_correction(args.bias_corr) \
                                     .with_calibration(CalibrationMethod.from_str(args.calib_method)) \
                                     .with_channel_equalization(args.chan_equal) \
                                     .with_smooth_quant(False)
  
  # quantize
  quant_model = loaded_net.quantize(calibration_data=length_hinted(len(calib_data),calib_data),
                                    quantization_config=quant_config,
                                    model_name=output_model_name,
                                    log_level=logging.INFO)

  # optional save of quantized model - saved model can be opened with Netron
  quant_model.save(model_name=output_model_name, output_directory=results_dir)
  print(f'Quantized model saved to {results_dir}/{output_model_name}.sima.json',flush=True)



  '''
  Execute, evaluate quantized model
  '''
  if (args.evaluate):

    annotated_images = (build_dir / 'quant_pred').resolve()

    utils.prepare_output_dir(f'{annotated_images}')
    print(f"Annotated images will be written to {annotated_images}", flush=True)

    image_paths = utils.get_image_paths(args.test_dir)
    num_images = min(args.num_test_images, len(image_paths))
    print(f'Using {num_images} out of {len(image_paths)}  test images',flush=True)
    image_paths = image_paths[:num_images]

    inputs = dict()
    for input_name in input_shapes_dict.keys():
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            print(f"Processing image: {filename}", flush=True)

            # Load original image (any size)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"  WARNING: Could not read image, skipping: {img_path}")
                continue

            orig_h, orig_w = img_bgr.shape[:2]

            inputs[input_name] = utils.preprocess_image(img_bgr,do_transpose=False)

            '''
            Returns a list of np arrays
            (1, 160, 160, 64)
            (1, 80, 80, 64)
            (1, 40, 40, 64)
            (1, 20, 20, 64)
            (1, 160, 160, 80)
            (1, 80, 80, 80)
            (1, 40, 40, 80)
            (1, 20, 20, 80)
            '''
            quantized_net_output = quant_model.execute(inputs, fast_mode=True)

            for i, out in enumerate(quantized_net_output):
                quantized_net_output[i] = np.transpose(out, (0, 3, 1, 2))


            # Postprocess in 640x640 space
            boxes_640, scores, class_ids = utils.postprocess_yolov8x_p2_4o(
                quantized_net_output,
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


            out_path = os.path.join(annotated_images, filename)
            ok = cv2.imwrite(out_path, annotated)
            if not ok:
                raise RuntimeError(f"Failed to write output image: {out_path}")
            print(f"  Annotated image written to: {out_path}")


  '''
  Compile
  '''
  print(f'Compiling with batch size set to {args.batch_size}',flush=True)
  quant_model.compile(output_path=results_dir,
                      batch_size=args.batch_size,
                      log_level=logging.WARN)  

  print(f'Wrote compiled model to {results_dir}/{output_model_name}_mpk.tar.gz',flush=True)

  # extract elf anfd mpk json for use in bechmarking
  with tarfile.open(f'{results_dir}/{output_model_name}_mpk.tar.gz') as tar:
     tar.extract(f'{output_model_name}_mpk.json' ,f'{results_dir}/benchmark')
     tar.extract(f'{output_model_name}_stage1_mla.elf' ,f'{results_dir}/benchmark')


  return



def run_main():
  
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-bd',  '--build_dir',         type=str, default='build', help='Path of build folder. Default is build')
  ap.add_argument('-m',   '--model_path',        type=str, default='yolov8x-p2_opt_4o.onnx', help='path to FP model')
  ap.add_argument('-b',   '--batch_size',        type=int, default=1, help='requested batch size. Default is 1')
  ap.add_argument('-om',  '--output_model_name', type=str, default='yolov8x-p2_opt_4o', help="Output model name. Default is yolov8x-p2_opt_4o")
  ap.add_argument('-cd',  '--calib_dir',         type=str, default='./calib_images', help='Path to calib images folder. Default is ./calib_images')
  ap.add_argument('-td',  '--test_dir',          type=str, default='./test_images', help='Path to test images folder. Default is ./test_images')
  ap.add_argument('-ci',  '--num_calib_images',  type=int, default=100, help='Number of calibration images. Default is 100')
  ap.add_argument('-ti',  '--num_test_images',   type=int, default=10, help='Number of test images. Default is 10')
  ap.add_argument('-g',   '--generation',        type=int, default=2, choices=[1,2], help='Target device: 1 = DaVinci, 2 = Modalix. Default is 2')
  ap.add_argument('-e',   '--evaluate',          action="store_true", default=False, help="If set, evaluate the quantized model") 
  ap.add_argument('-cm',  '--calib_method',      type=str, default='min_max', choices=['mse','min_max','moving_average','entropy','percentile'], help="Calibration method. Default is min_max") 
  ap.add_argument('-qb',  '--quant_bits',        type=int, default=8, choices=[8,16], help="Bits for quantization. Default is 8bits")
  ap.add_argument('-bc',  '--bias_corr',         action='store_true', help="Use bias correction. Default is no bias correction")
  ap.add_argument('-ce',  '--chan_equal',        action='store_true', help="Use channel equalization. Default is no channel equalization")
  ap.add_argument('-bf16', '--bf16',             action='store_true', help="Use BlockFloat16 quantization. If not set, quant_bits argument is used")
  ap.add_argument('-ct',  '--conf_thres',        type=float, default=0.45, help="Confidence threshold")
  ap.add_argument('-it',  '--iou_thres',         type=float, default=0.45, help="IoU threshold for NMS")
  args = ap.parse_args()

  print('\n'+DIVIDER,flush=True)
  print('Model SDK version',get_model_sdk_version())
  print(sys.version,flush=True)
  print(DIVIDER,flush=True)


  implement(args)


if __name__ == '__main__':
    run_main()