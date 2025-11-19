#########################################################
# Copyright (C) 2023 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
# Code owner: Ashok Sudarsanam
#########################################################
"""
Given a network which produces a combination of .elf and .so
files when compiled, run these files on the DevKit and emit
FPS KPIs to the console.
"""
import os
import argparse
from os import path
from typing import List, Dict
import time
import json
import warnings
import numpy as np
import sys
import os
import logging
import getpass
import json
import subprocess
import signal    # Import signal module for handling Ctrl+C/Ctrl+Z

# Determine project root directory for module imports
PROJECT_ROOT = os.path.abspath(sys.path[0].rsplit('/',1)[0])
sys.path.append(PROJECT_ROOT)
MLA_ONLY_MODE = True  # Flag to determine if running in MLA-only mode or MLA+A65 mode

#suppress warnings
warnings.filterwarnings('ignore')

import utils
import apis

from ev_transforms.transforms import (
    quantize, tessellation,
)

from mpk_attributes import MPKAttrs
from apis.mla_rt_client import Mla_rpc_client as mla_rt_client
from apis.mla_rt_client import DEFAULT_ZMQ_PORT
from utils.ssh_utils import create_forward_tunnel, create_forward_tunnel_generic, scp_files_to_remote, run_remote_command, get_layer_stats_yaml_file

# Constants for DevKit connection
DEVKIT_FOLDER="/home/sima"
DEFAULT_PORT=8000
DEFAULT_USERNAME='sima'

class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.fcounter = 0  # Frame counter for tracking processed frames
        self.report_fcount = 10 # update stats every 10 frames
        self.inference_time_accum = 0  # Accumulator for inference timing
        self.avg_fps = 0.0  # Average FPS calculation
        self.ifm_arr = []  # Array to hold input feature maps
        self.ofm_arr = []  # Array to hold output feature maps
        self.ifm = None    # Single input feature map reference
        self.ofm = None    # Single output feature map reference
        self.batch_size = args.batch_size  # Batch size for inference

        # Parse MPK JSON file to get model attributes
        self.mpk_dict = Pipeline.get_mpk_json(args.mpk_json_path)
        if not any(self.mpk_dict.values()):
            raise RuntimeError("ERROR: MPK dict is empty")

        # Initialize RPC client for DevKit communication
        self.rpc_client = mla_rt_client(args.dv_host, args.dv_port, not MLA_ONLY_MODE)
        if self.rpc_client.connect() is False:
            raise RuntimeError(f"ERROR: MLA RPYC connect to DevKit {args.dv_host} failed")

        if MLA_ONLY_MODE:
            # Load the MPK attributes for MLA-only mode
            self.mpk_attrs = MPKAttrs(self.mpk_dict)

            # Validate batch size matches model requirements
            if self.batch_size != self.mpk_attrs.batch_size:
                raise RuntimeError(f"Specified batch size {self.batch_size} does not match model batch size {self.mpk_attrs.batch_size}")

            # Connect to DevKit and load the .elf model file
            if self.rpc_client.dv_connect(args.layer_stats_path) is not True:
                raise RuntimeError(f"ERROR: Cannot connect to DevKit")
            model_file_path = os.path.join(DEVKIT_FOLDER, os.path.basename(self.args.model_file_path))
            self.model = self.rpc_client.load_model_only(model_file_path)

            # Adjust input/output feature map sizes for batched models
            self.ifm_size = int(self.mpk_attrs.ifm_size/self.mpk_attrs.batch_size)
            self.ofm_size = int(self.mpk_attrs.ofm_size/self.mpk_attrs.batch_size)

            # Allocate input and output feature maps for each batch element
            for i in range(self.batch_size):
                ifm = self.rpc_client.allocate_ifm(self.ifm_size)
                if not ifm:
                    raise RuntimeError(f"allocate_ifm failed")
                self.ifm_arr.append(ifm)
                ofm = self.rpc_client.allocate_ofm(self.ofm_size)
                if not ofm:
                    raise RuntimeError(f"allocate_ofm failed")
                self.ofm_arr.append(ofm)
            # Save references to first ifm and ofm for upload/download operations
            self.ifm = self.ifm_arr[0]
            self.ofm = self.ofm_arr[0]
        else:
            # Connect to DevKit and load .tar.gz file for MLA+A65 mode
            if self.rpc_client.pm_connect(args.model_file_path) is not True:
                raise RuntimeError(f"ERROR: Cannot connect to DevKit")

    @staticmethod
    def get_mpk_json(mpk_json_path) -> Dict:
        """
        Load the json file into a dictionary.

        :param mpk_json_path: the fullpath for mpk_json_path
        :return: a dict.
        """
        mpk_dict = {}

        try:
            # Load JSON configuration file
            with open(mpk_json_path) as json_file:
                mpk_dict = json.load(json_file)
        except ValueError:
            return None

        return mpk_dict

    def run_inference(self, prep_image: np.ndarray, fcounter: int) -> bytearray:
        """
        Run the inference model
        :param prep_image: a formatted RGB image.
        :param fcounter: running frame counter
        :return: pass or fail
        """            
        # Execute model inference on DevKit with batched input/output
        rc, run_time = self.rpc_client.run_batch_model_phys(self.model, self.batch_size, self.ifm_arr, self.ifm_size, self.ofm_arr, self.ofm_size)
        if rc:
            # Handle inference failure cases
            if self.batch_size != self.mpk_attrs.batch_size:
                raise RuntimeError(f"Running model failed - specified batch size {self.batch_size} does not match model batch size {self.mpk_attrs.batch_size}")
            else:
                raise RuntimeError(f"Running model failed")
        
        # Convert runtime from microseconds to seconds
        run_time_in_secs = run_time/1000000
        self.inference_time_accum += run_time_in_secs
        self.fcounter += self.batch_size
        logging.debug("Total Time to run frame %s : %.3f", fcounter, run_time_in_secs)
        self.update_inference_fps()

        return 0

    def run_pipelined_inference(self, preprocessed_frame: np.ndarray, fcounter: int) -> np.ndarray:
        """
        Run the inference model in pipeline mode which could be
        mix of A65 and MLA execution
        :param preprocessed_frame: a tessellation transform.
        :param fcounter: running frame counter
        :return: ofm_np_array
        """
        # Execute pipelined inference (MLA+A65 mode)
        time_start = time.time()
        ofm_np_array = self.rpc_client.pm_run_pipeline([preprocessed_frame])
        time_end = time.time()
        run_time = time_end - time_start
        self.inference_time_accum += run_time
        self.fcounter += 1
        logging.debug("Total Time to run frame %s : %.3f", fcounter, run_time)
        self.update_inference_fps()
        
        return ofm_np_array

    def release(self) -> None:
        """
        Free model and allocated memory resources
        """
        if MLA_ONLY_MODE:
            # Free allocated feature maps and model for MLA-only mode
            self.rpc_client.free_ifm(self.ifm)
            self.rpc_client.free_ofm(self.ofm)
            self.rpc_client.free_model(self.model)
        else:
            # Free resources for pipelined mode
            self.rpc_client.pm_free_frames()
            self.rpc_client.pm_free_model()

    def update_inference_fps(self):
        """
        Update and emit the FPS KPIs every 10 frames.
        """
        if self.fcounter >= self.report_fcount:
            # Calculate and display FPS metrics
            if self.inference_time_accum != 0:
                avg_inference_time = self.inference_time_accum / self.fcounter
                self.avg_fps = 1.0 / avg_inference_time
            else:
                self.avg_fps = 0.0
            self.fcounter = 0
            self.inference_time_accum = 0

            print(f"FPS = {int(self.avg_fps)}")

    def signal_handler(self, sign_num, frame):
        """
        Handle the case of getting Ctrl + C
        """
        print("Ctrl + C is pressed.")
        self.release()  # Clean up resources before exiting
        sys.exit(0)

    def signal_Z_handler(self, sign_num, frame):
        """
        Handle the case of getting Ctrl + Z
        """
        print("Ctrl + Z is pressed.")
        self.release()  # Clean up resources before exiting
        sys.exit(0)

            
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Emit FPS KPIs for networks that run on the MLA and/or A65')
    parser.add_argument('--model_file_path', required=True, type=str, help='Path to .elf or .tar.gz file')
    parser.add_argument('--mpk_json_path', required=True, type=str, help='Path to MPK JSON file')
    parser.add_argument('--dv_host', required=True, type=str, help='DevKit IP Address / FQDN')
    parser.add_argument('--dv_port', required=False, type=int, default=DEFAULT_PORT, help='DevKit port on which the mla_rt_service is running')
    parser.add_argument('--dv_user', required=False, type=str, default=DEFAULT_USERNAME, help='DevKit ssh username')
    parser.add_argument('--image_size', required=True, nargs='+', type=int, help='RGB image size specified as: H W C')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument('--bypass_tunnel', required=False, action='store_true', help= 'set to bypass ssh tunnel')
    parser.add_argument('--layer_stats_path', required=False, type=str, help='Path to layer stats YAML file')
    parser.add_argument('--max_frames', required=False, type=int, default=0, help='Max number of frames to run')
    parser.add_argument('--batch_size', required=False, type=int, default=1, help='Batch size - default 1')
    
    args = parser.parse_args()
    
    # Determine execution mode based on model file extension
    if args.model_file_path.endswith(".elf"):
        print("Running model in MLA-only mode")
    else:
        MLA_ONLY_MODE = False
        print("Running model in MLA+A65 mode")

    # Configure logging if verbose mode is enabled
    if args.verbose:
        # Create the logfile with BasicConfig
        # logging.basicConfig(filename='network_eval.log', encoding='utf-8', level=logging.DEBUG)
        logging.basicConfig(filename='network_eval.log', level=logging.DEBUG)
        logging.debug('This message should go to the log file')

    # Extract image dimensions from command line arguments
    H, W, C = args.image_size
    password = ''
    max_attempts = 10
    
    # Set up SSH tunnel if not bypassed
    if not args.bypass_tunnel:
        ssh_connection, local_port = create_forward_tunnel(args, password, max_attempts)
        
        if ssh_connection is None:
            logging.debug(f'Failed to forward local port after {max_attempts}')
            sys.exit(-1)

        # Create tunnel for ZMQ client/server for large data transfers
        zmq_ssh_connection, zmq_local_port = create_forward_tunnel_generic(DEFAULT_ZMQ_PORT, DEFAULT_USERNAME, args.dv_host, password, max_attempts)

        if zmq_ssh_connection is None:
            print(f'Failed to forward zmq local port after {max_attempts}')
            sys.exit(-1)

        # Update port to use local tunneled port
        args.dv_port = local_port

    # Copy model file (.elf or .tar.gz) to the DevKit
    scp_file = scp_files_to_remote(args, args.model_file_path, password, DEVKIT_FOLDER, max_attempts)
    if scp_file is None:
        logging.error(f'Failed to scp the model file after {max_attempts}')
        sys.exit(-1)

    # Validate layer stats is only used with MLA-only mode
    if not MLA_ONLY_MODE and args.layer_stats_path:
        logging.error("Layer stats only supported with MLA Only mode")
        sys.exit(-1)

    # Copy layer stats YAML file if provided
    if args.layer_stats_path:
        scp_file = scp_files_to_remote(args, args.layer_stats_path, password, DEVKIT_FOLDER, max_attempts)
        if scp_file is None:
            logging.error(f'Failed to scp the layer file after {max_attempts}')
            sys.exit(-1)

    # Initialize pipeline for inference
    pipeline = Pipeline(args)

    fcounter = 1
    # Main inference loop
    while True:
        # Generate random input frame for testing
        frame = np.random.rand(1, H, W, C).astype(np.float32)
        if MLA_ONLY_MODE:
            pipeline.run_inference(prep_image=frame, fcounter=fcounter)
            
            # Copy back layer stats file after first frame (if enabled)
            if args.layer_stats_path:
                if fcounter == 1:
                    print("Copying output layer stats yaml file - enter password")
                    get_layer_stats_yaml_file(args, password, args.layer_stats_path, max_attempts)
                    break
                    
        else:
            # Run pipelined inference for MLA+A65 mode
            pipeline.run_pipelined_inference(preprocessed_frame=frame, fcounter=fcounter)
            
        # Check if maximum frame limit reached
        if args.max_frames == fcounter:
            print(f"Ran {fcounter} frame(s)")
            break

        fcounter = fcounter + 1
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, pipeline.signal_handler)
        signal.signal(signal.SIGTSTP, pipeline.signal_Z_handler)

    # Clean up resources
    pipeline.release()
    
    # Close SSH tunnel if it was created
    if not args.bypass_tunnel:
        ssh_connection.kill()
