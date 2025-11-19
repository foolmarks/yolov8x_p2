#########################################################
# Copyright (C) 2024 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
# Code owner: Nevena Stojnic
#########################################################
"""
This API provides user to create Pipeline class for accelerator mode that
is used to acquire connection with the devkit and to run inference on the
devkit. If user needs to perform pre processing of the data before running
inference and post processing of the data after running inference they should
use class PrePostProcessingPipeline, as this class is designed for that use.

This API can be installed and used in other projects. For example:
    cd devkit-inference-examples
    python3 setup.py bdist_wheel (or bash scripts/build_scripts/build_py_pkg.sh)
    scp dist/sima_accelerator_mode-2.0.0-py3-none-any.whl <project_name>
    cd <project_name>
    pip3 install sima_accelerator_mode-2.0.0-py3-none-any.whl

Package name is sima_accelerator_mode, and functions and classes from
this API should be included from that package. For example:
    from devkit_inference_models.apis.pipeline include Pipeline

"""

import os
from os import path
import sys
from typing import List, Dict, Any, Union, Tuple, Callable, Optional
import time
import warnings

import numpy as np
import json

from devkit_inference_models.apis.mla_rt_client import Mla_rpc_client
from ev_transforms.transforms import *

PROJECT_ROOT = os.path.abspath(sys.path[0].rsplit('/', 1)[0])
sys.path.append(PROJECT_ROOT)

# suppress warnings
warnings.filterwarnings('ignore')

DEVKIT_FOLDER = "/home/sima"


# Dict to hold mapping kernels to ev_transform functions
_EV_TRANSFORM_FROM_KERNEL: Dict[str, Tuple[Callable, List[str]]] = {
    "tessellation_transform":   (tessellation,
                                 ["data", "slice_height", "slice_width",
                                  "slice_channels", "align_c16"]),
    "detessellation_transform": (detessellation,
                                 ["data", "slice_height", "slice_width",
                                  "slice_channels", "frame_type",
                                  "frame_shape"]),
    "quantization_transform":   (quantize,
                                 ["frames", "channel_params", "num_bits"]),
    "dequantization_transform": (dequantize,
                                 ["q_frames", "channel_params"]),
    "reshape_transform":        (reshape,
                                 ["newshape", "data"]),
    "unpack_transform":         (unpack,
                                 ["data", "tensor_types", "tensor_shapes"]),
    "layout_transform":         (layout_transform,
                                 ["data", "src_layout", "dst_layout"]),
}


class Pipeline:
    """ This class is responsible for:
        1. Connecting to the DevKit.
        2. Storing IFM into the DevKit
        3. Running the inference on the DevKit.
        4. Downloading OFM output from the DevKit.
        5. Converting output into numpy array.
    """
    def __init__(self, dv_host: str, dv_port: str,
                 model_file_path: str) -> None:

        assert dv_host is not None, "Please provide hostname."
        assert dv_port is not None, "Please provide port."
        assert model_file_path is not None, "Please provide model lm file path."

        self.rpc_client = Mla_rpc_client(dv_host, dv_port, False)
        if self.rpc_client.connect() is False:
            raise RuntimeError("ERROR: MLA RPYC connect failed")

        if self.rpc_client.dv_connect() is not True:
            raise RuntimeError(f"ERROR: Cannot connect to DevKit")

        # Load the LM file on the DevKit
        if (model_file_path):
            model_file_path = os.path.join(DEVKIT_FOLDER, os.path.basename(model_file_path))
            self.model = self.rpc_client.load_model_only(model_file_path)
        else:
            raise RuntimeError(f"ERROR: {model_file_path} LM file does not exist")

        self.ifm = None
        self.ofm = None

    def is_ifm_ofm_initialized(self) -> bool:
        """
        Check if IFM and OFM size parameters are
        initialized. IFM and OFM initialization
        (that includes allocation on the DevKit) happens
        only once (before running first inference on the
        DevKit). This function is used to check if the
        allocation is already performed.
        :return: bool value that indicates if IFM and OFM
        are allocated or not.
        """
        if not self.ifm or not self.ofm:
            return False
        return True

    def init_ifm_ofm(self, ifm_size, ofm_size) -> None:
        """
        Save IFM and OFM size and allocate IFM and OFM
        on the DevKit
        :param ifm_size: integer size of IFM
        :param ofm_size: integer size of OFM
        """
        self.ifm_size = ifm_size
        self.ofm_size = ofm_size

        # Allocate ofm once during the initialization
        self.ifm = self.rpc_client.allocate_ifm(self.ifm_size)
        if not self.ifm:
            raise RuntimeError(f"allocate_ifm failed")
        self.ofm = self.rpc_client.allocate_ofm(self.ofm_size)
        if not self.ofm:
            # Free IFM to avoid memory leak
            self.rpc_client.free_ifm(self.ifm)
            raise RuntimeError(f"allocate_ofm failed")

    def _upload_preprocess_file(self, tessed_out: np.ndarray):
        """
        Write the output of pre_process image to the raw binary file,
        and upload to DevKit.
        :param tessed_out: a tessellation transform.
        :return: an obj mla_rt_handle
        """
        return self.rpc_client.upload_ifm_zmq(self.ifm, tessed_out.tobytes())

    def run_inference(self, preprocessed_frame: np.ndarray) -> np.ndarray:
        """
        Run the inference model
        :param preprocessed_frame: a formatted RGB image (a
        tessellation transform).
        :param fcounter: running frame counter
        :return: ofm_bytes, upload_time, run_time, download_time
        """

        # Allocate and upload ifm to the card allocated memory
        start_time = time.time()
        self._upload_preprocess_file(preprocessed_frame)
        end_time = time.time()
        upload_time = end_time - start_time

        # Run the model on the card with above model, ifm and ofm
        rc, run_time = self.rpc_client.run_model_phys(self.model, self.ifm, self.ofm)
        if rc:
            raise RuntimeError(f"Running model failed")

        # Download ofm from card into local byte array
        start_time = time.time()
        ofm_bytes = self.rpc_client.download_ofm_only_zmq(self.ofm, self.ofm_size)
        end_time = time.time()
        download_time = end_time - start_time
        return ofm_bytes, upload_time, run_time, download_time

    def _convert_ofm_data(self, ofm_bytes) -> np.ndarray:
        """
        Convert ofm_bytes to numpy array of output nodes.
        :param file_name : ofm_bytes.
        :return: output nodes.
        """
        output_node = np.frombuffer(ofm_bytes, dtype=np.int8, offset=0, count=self.ofm_size)

        if output_node.size == 0:
            raise RuntimeError(f"convert_ofm_data: Expect an np.ndarray, got empty np.ndarray")

        return output_node

    def postprocess_output(self, ofm_bytes) -> np.ndarray:
        """
        Do post-process just by converting ofm_bytes into
        numpy array.
        :param ofm_bytes: a ofm_bytes.
        :return: output from the DevKit in numpy array.
        """
        out_data = self._convert_ofm_data(ofm_bytes)

        return out_data

    def release(self) -> None:
        """
        Free model
        """
        if self.ifm:
            self.rpc_client.free_ifm(self.ifm)

        if self.ofm:
            self.rpc_client.free_ofm(self.ofm)

        if self.model:
            self.rpc_client.free_model(self.model)

        self.rpc_client.zmq_socket.close()
        self.rpc_client.zmq_context.term()


class PrePostProcessingPipeline(Pipeline):
    """ This class is responsible for:
        1. Connecting to the DevKit.
        2. Reading mpk.json file and taking information from the
           preprocessing and postprocessing plugins.
        3. Preprocessing input into the model in a way that it can be
           stored on the DevKit.
        4. Storing IFM into the DevKit
        5. Running the inference on the DevKit.
        6. Downloading OFM output from the DevKit.
        7. Converting output into numpy array.
        8. Postprocessing output of the model.
    """
    def __init__(self, dv_host, dv_port, model_file_path, mpk_json_path) -> None:

        super().__init__(dv_host, dv_port, model_file_path)
        assert mpk_json_path is not None, "Please provide model mpk.json file path."

        # Parse MPK JSON
        self.mpk_dict = PrePostProcessingPipeline._get_mpk_json(mpk_json_path)
        if not any(self.mpk_dict.values()):
            raise RuntimeError("ERROR: MPK dict is empty")

        # Load the MPK attributes
        plugins = self.mpk_dict["plugins"]
        self.preprocess_plugins = []
        self.postprocess_plugins = []
        mla_plugin_found = False
        for plugin in plugins:
            processor_type = plugin["processor"]
            if processor_type == "EV74":
                if mla_plugin_found:
                    self.postprocess_plugins.append(plugin)
                else:
                    self.preprocess_plugins.append(plugin)
            elif processor_type == "MLA":
                ifm_size = plugin["input_nodes"][0]["size"]
                ofm_size = plugin["output_nodes"][0]["size"]
                mla_plugin_found = True

            else:
                raise RuntimeError(f"ERROR: Unsupported processor type: {processor_type}")

        super().init_ifm_ofm(ifm_size, ofm_size)

    @staticmethod
    def _get_mpk_json(mpk_json_path) -> Dict:
        """
        Load the json file into a dictionary.

        :param mpk_json_path: the fullpath for mpk_json_path
        :return: a dict.
        """
        mpk_dict = {}

        try:
            with open(mpk_json_path) as json_file:
                mpk_dict = json.load(json_file)
        except ValueError:
            return None

        return mpk_dict

    def _run_ev_kernel(self, kernel: str, params: Dict[str, Any],
                      data: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Call appropriate ev_transform function.
        :param kernel: ev_transform kernel name that should be called.
        :param params: parameters needed for the ev_transform function.
        :param data: input data to the ev_transform function.
        :return: output from the ev_transform function.
        """
        if (kernel in _EV_TRANSFORM_FROM_KERNEL):
            transform_fn, param_names = _EV_TRANSFORM_FROM_KERNEL[kernel]
            transform_params: Dict[str, Any] = {}

            for param in param_names:
                if param in ['frames', 'q_frames', 'data']:
                    # Convert the 4D input (pack transform data does not have ndim)
                    if (((hasattr(data, 'ndim')) and (data.ndim == 3))
                            or (kernel == "unpack_transform")):
                        data = np.expand_dims(data, axis=0)
                    transform_params[param] = data
                    data_shape = data[0].shape if isinstance(data, List) else data.shape
                else:
                    assert param in params.keys()
                    transform_params[param] = params[param]
            try:
                out = transform_fn(**transform_params)
            except Exception as err:
                data_shape = data[0].shape if isinstance(data, List) else data.shape
                err_msg = "\nException: " + str(err) if len(err.args) < 1 else str(err.args[0])
                info_msg = "\nFailed to execute {}, with params {}, \
                            data {} shape {}".format(kernel, params, type(data), data_shape)
                raise Exception(err_msg + info_msg)
            return out
        else:
            raise Exception(f"\nException: {kernel} is not implemented yet.")

    def image_preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Pre-process the image (described in mpk.json file of the model).
        :param frame: an RGB image
        :return: The tessellation transform (output tensor is 1D array).
        """
        input_frame = frame

        for plugin in self.preprocess_plugins:
            config_params = plugin["config_params"]
            ev_kernel = config_params['kernel']
            out_frame = self._run_ev_kernel(ev_kernel, config_params["params"], input_frame)
            input_frame = out_frame

        return out_frame

    def postprocess_output(self, ofm_bytes) -> int:
        """
        Do post-process for each output node in the list.
        Post-process is performed by reading through all of the plugins
        from mpk.json file of the model and calling appropriate
        ev_transform functions.
        :param ofm_bytes: a ofm_bytes.
        :return: model output.
        """

        out_data = super().postprocess_output(ofm_bytes)

        input_frame = out_data

        # This is flag that annotates that the input into the
        # ev_transform function should be directly output of the model
        # (out_data should be used).
        processing_mla_output = True

        output = [0]
        idx = 0

        # This dictionary keeps track of the names of all of the
        # remaining nodes from mpk.json file, and links their names
        # with the index of the appropriate output from the list of all
        # outputs. That way we can perform ev transform from the
        # current plugin on the proper output.
        nodes = {}
        for plugin in self.postprocess_plugins:
            config_params = plugin["config_params"]
            ev_kernel = config_params['kernel']
            if ev_kernel == "pass_through":
                continue
            output_nodes = plugin["output_nodes"]
            input_nodes = plugin["input_nodes"]

            if processing_mla_output:
                processing_mla_output = False
            else:
                # Get the index of the right output from the list of
                # all outputs. That output is used as the input into
                # ev_transform defined in current plugin. Index is
                # aquired from nodes dictionary that links node names
                # with appropriate output indexes. After this transform
                # output with idx index will get a new name so there is
                # no need to keep the current name in the dictionary so
                # we can delete it. New name will be updated later.
                idx = nodes[input_nodes[0]['name']]
                del nodes[input_nodes[0]['name']]
                input_frame = output[idx]

            out_frame = self._run_ev_kernel(ev_kernel, config_params["params"], input_frame)
            assert len(output_nodes) == len(out_frame)

            # Only unpack_transform have multiple outputs
            if len(output_nodes) > 1:
                assert ev_kernel == "unpack_transform"
                output = out_frame
                # Update nodes dictionary with multiple outputs from
                # unpack transform and their indexes.
                i = 0
                for node in output_nodes:
                    nodes[node['name']] = i
                    i = i + 1
            else:
                output[idx] = out_frame
                # Update nodes dictionary with most recent name of the
                # output with idx index
                nodes[output_nodes[0]['name']] = idx

        return output[0] if len(output) == 1 else output
