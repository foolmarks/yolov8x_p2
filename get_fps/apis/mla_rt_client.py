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
# Code owner: Vimal Nakum
#########################################################
"""
This is remote procedure call (RPC) client.
"""
import rpyc
import argparse
import shutil
import os
import time
import ctypes
import numpy as np
import zmq

from utils.common_utils import check_tar_gz_file

MLA_RPC_CLIENT_VERSION=1.07
DEFAULT_ZMQ_PORT = 43777

class Mla_rpc_client:

    def __init__(self, host, port, pipeline):
        self.remote = None
        self.host = host
        self.port = port
        self.conn = None
        self.model = None
        self.ifm = None
        self.ofm = None
        self.pipeline = pipeline
        # This version has to match with the server version for compatibility
        return

    def connect(self):
        # Check if secure ssl connection is requested
        self.conn = rpyc.connect(self.host, self.port,config={'allow_public_attrs': True, "allow_pickle":True, "sync_request_timeout":None})
        if (not self.conn):
            raise Exception("[ ERROR ] Failed to connect to {}:{}",format(self.host, self.port))

        # test connection
        # if rpyc server already has an active connection, it would close the new
        # connection, which results in EOFError, so catch that error and return
        # error to the caller
        try:
            if (self.conn.root.ping("ping") != "pong"):
                raise Exception("[ ERROR ] Ping test failed")
        except EOFError:
            print(f"RPYC Connection failed as DevKit {self.host} is busy")
            return False
        # check versions
        rc = self.conn.root.check_version(MLA_RPC_CLIENT_VERSION)
        
        if (rc) is not True:
            raise Exception("[ ERROR ] Version check failed server version {} client version {}".format(rc, MLA_RPC_CLIENT_VERSION))

        return True

    def dv_connect(self, layer_stats_path=None):
        """
        connects to DV board using non-pipelined mode
        """
        # pass the layer file path to connect, which would
        # pass it to get_handle function eventually
        layer_file_path = None
        if layer_stats_path:
            layer_file_path = "/home/sima/" + os.path.basename(layer_stats_path)
        rc = self.conn.root.dv_connect(layer_file_path)
        if (rc) is not True:
            raise Exception("[ ERROR ] Connecting dv failed")
        
        # Context and socket using the local port forwarded by the tunnel
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PAIR)
        
        # would return exception if failed
        self.zmq_socket.connect(f"tcp://localhost:{DEFAULT_ZMQ_PORT}")
        return rc

    def pm_connect(self, tar_file_path):
        """
        connects to DV board using pipelined mode
        """        
        # keep only the file name
        tar_file_name = os.path.basename(tar_file_path)
        assert check_tar_gz_file(tar_file_name), f"{tar_file_name} must be in tar format."
        rc = self.conn.root.pm_connect(tar_file_name)
        if (rc) is not True:
                raise Exception("[ ERROR ] Connecting to DV Pipeline failed")
        return rc

    # load the model that is already present on the card
    def load_model_only(self, model_name):
        # Execute load model call on the server
        self.model = self.conn.root.load_model(model_name)
        if not self.model:
            raise Exception("[ ERROR ] Load model failed")
        return self.model

    # allocate ifm, later use upload_ifm to upload data into this ifm
    def allocate_ifm(self, ifm_len):
        if not ifm_len:
            logger.error("invalid ifm length 0")
            return None

        self.ifm = self.conn.root.allocate_ifm(ifm_len)
        return self.ifm

    def allocate_array_ifm(self, ifm_len, array_size: int=2):
        """
        Function to allocare array of IFM.
        :params ifm_len: the len of ifm
        :params array_size: the size of IFM array

        return: the array of virtual memory buffer IFM if successful
        """
        if not ifm_len:
            logger.error("invalid ifm length 0")
            return None

        return self.conn.root.allocate_array_ifm(ifm_len, array_size)

    def upload_array_ifm(self, arr_mem_buff_ifm, array_ifm_data):
        """
        Function to upload the array of IFM data to the card
        :params arr_mem_buff_ifm: the array of virtual memory buffer IFM
        :params array_ifm_data: the data of its array IFM (needs to be in bytes)

        return: True if successful to upload IFM.
        """
        if not arr_mem_buff_ifm or not array_ifm_data:
            logger.error("Invalid the array of IFM or the array of IFM data")
            return False

        rc = self.conn.root.upload_array_ifm(arr_mem_buff_ifm, array_ifm_data)
        return rc

    # upload the provided ifm_data to the card allocated ifm buffer
    # ifm_data needs to be in bytes
    def upload_ifm(self, ifm, ifm_data):
        if not ifm or not ifm_data:
            logger.error("invalid ifm or ifm_data")
            return False

        rc = self.conn.root.upload_ifm(ifm, ifm_data)
        return rc

    # use zmq to upload ifm data, as it is more efficient than
    # using rpyc
    def upload_ifm_zmq(self, ifm, ifm_data):
        if not ifm or not ifm_data:
            logger.error("invalid ifm or ifm_data")
            return False

        self.zmq_socket.send(ifm_data)
        rc = self.conn.root.upload_ifm_zmq(ifm, len(ifm_data))
        return rc

    # allocate OFM on the card
    def allocate_ofm(self, ofm_len):
        if not ofm_len:
            logger.error("invalid ofm length 0")
            return None
        self.ofm = self.conn.root.allocate_ofm(ofm_len)
        return self.ofm

    def allocate_array_ofm(self, ofm_len, array_size: int=1):
        """
        Function to allocate the array of OFM virtual memory.
        :params ofm_len: the length of OFM.
        :params array_size: the size of array of virtual memory buffer OFM.

        return: True, pointer of the array of virtual memory buffer OFM.
        """
        if not ofm_len:
            logger.error("invalid ofm length 0")
            return False, None

        return self.conn.root.allocate_array_ofm(ofm_len, array_size)

    # download ofm data from the card and return as bytes
    def download_ofm_only(self, ofm, ofm_len):
        if not ofm_len:
            logger.error("invalid ofm length 0")
            return None
        # copy the ofm data from model result on the ard
        ofm_bytes = self.conn.root.download_ofm(ofm, ofm_len)
        return ofm_bytes

    def download_array_ofm(self, arr_mem_buff_ofm, ofm_len):
        """
        Function to download the array of OFM.
        :params arr_mem_buff_ofm: the array of virtual memory buffer OFM.
        :params ofm_len: the length of OFM.

        return: True, the array of OFM data.
        """
        # download ofm data from the card and return as bytes
        if not arr_mem_buff_ofm or not ofm_len:
            logger.error("Invalid the array of OFM or the length of OFM")
            return False, None
        rc, array_ofm_buff = self.conn.root.download_array_ofm(arr_mem_buff_ofm, ofm_len)
        if rc:
            return True, array_ofm_buff


    # download ofm data from the devkit and return as bytes
    # use zmq to download data as it is more efficient than
    # using rpyc
    def download_ofm_only_zmq(self, ofm, ofm_len):
        if not ofm_len:
            logger.error("invalid ofm length 0")
            return None
        # copy the ofm data from model result on the devkit
        self.conn.root.download_ofm_zmq(ofm, ofm_len)
        ofm_bytes = self.zmq_socket.recv()
        return ofm_bytes
    
    def free_ifm(self, ifm):
        if ifm:
            self.conn.root.free_ifm(ifm)
        return

    def free_array_ifm(self, arr_mem_buff_ifm):
        """
        Function to free the array of IFM virtual memory buffer
        """
        if arr_mem_buff_ifm:
            self.conn.root.free_array_ifm(arr_mem_buff_ifm)
        return

    def free_array_ofm(self, arr_mem_buff_ofm):
        """
        Function to free the array of OFM virtual memory buffer
        """
        if arr_mem_buff_ofm:
            self.conn.root.free_array_ofm(arr_mem_buff_ofm)
        return

    def free_ofm(self, ofm):
        if ofm:
            self.conn.root.free_ofm(ofm)
        return

    # run the model
    def run_model_phys(self, model, ifm, ofm):
        if model and ifm and ofm:
            rc, run_time = self.conn.root.run_model_phys(model, ifm, ofm)
            return rc, run_time
        else:
            return -1, 0

    # run the batch model
    def run_batch_model_phys(self, model, batch_size, arr_mem_buff_ifm, ifm_size, arr_mem_buff_ofm, ofm_size):
        if model and arr_mem_buff_ifm and arr_mem_buff_ofm and batch_size and ifm_size and ofm_size:
            rc, run_time = self.conn.root.run_batch_model_phys(model, batch_size, arr_mem_buff_ifm, ifm_size, arr_mem_buff_ofm, ofm_size)
            return rc, run_time
        else:
            return -1, 0

    # free model
    def free_model(self, model):
        if model:
            self.conn.root.free_model(model)
        return

    # create virtual env
    def create_virtual_env(self, virtual_env_name):
        rc, result = self.conn.root.create_virtual_env(virtual_env_name)
        return rc, result

    # create virtual env
    def delete_virtual_env(self, virtual_env_name):
        rc = self.conn.root.delete_virtual_env(virtual_env_name)
        return rc

    # run remote model script
    def run_remote_model(self, full_command, model_file_path, timeout, virtual_env_name):
        rc, output, error = self.conn.root.run_remote_model(full_command, model_file_path, timeout, virtual_env_name)
        return rc, output, error

    # a65/mla pipeline related functions
    def pm_run_pipeline(self, ifm):
        #print(f"ifm = {ifm}")
        ofm_raw = self.conn.root.pm_run_pipeline(ifm)
        #print(f"ofm_raw = {ofm_raw}")
        if ofm_raw is not None:
            # by default rpyc does not do deep copy of the np array
            # use obtain call to get the entire np array
            ofm_np_array = rpyc.classic.obtain(ofm_raw)
            return ofm_np_array
        else:
            return None

    def pm_free_model(self):
        return self.conn.root.pm_free_model()
    
    def pm_free_frames(self):
        return self.conn.root.pm_free_frames()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Set Verbose", action="store_true", default=False)
    parser.add_argument("-d", "--debug", help="Set Debug")
    parser.add_argument("--host", help="Hostname (default localhost)", default=None)
    parser.add_argument("--port", help="Port (default 1000)", default=None)
    parser.add_argument("--keyfile", help="SSL Server Key file", default=None)
    parser.add_argument("--certfile", help="SSL Server Cert file", default=None)
    return parser.parse_args()

'''
if __name__ == '__main__':
    options = parse_arguments()
    port = 8000
    if options.port:
        port = options.port
    #rpc_client.load_firmware("mla_driver.bin")
    # load reid model and allocate their ifm/ofm
    #reid_model = rpc_client.load_model_only("onnx_ga_reid_batch24_dcmp.lm")
    #reid_ifm_file_name = "onnx_ga_reid_batch24_ifm.0.mlc"
    #reid_ofm_file_name = "onnx_ga_reid_batch24_ofm.0.mlc"
    #with open(reid_ifm_file_name, mode='rb') as ifm_file:
    #    # read file returns file data as bytes[]
    #    reid_ifm_file_contents = ifm_file.read()

    #reid_ifm = rpc_client.load_ifm_only(reid_ifm_file_contents, os.path.getsize(reid_ifm_file_name))
    #reid_ofm = rpc_client.allocate_ofm(os.path.getsize(reid_ofm_file_name))

    # load runway model and allocate their ifm/ofm
    #runway_model = rpc_client.load_model_only("onnx_ga_runway_centernet_asym-True_per_channel-True_dma-4_mb_dcmp.lm")
    #runway_ifm_file_name = "onnx_ga_runway_centernet_asym-True_per_channel-True_dma-4_mb_ifm.0.mlc"
    #runway_ofm_file_name = "onnx_ga_runway_centernet_asym-True_per_channel-True_dma-4_mb_ofm.0.mlc"
    #with open(runway_ifm_file_name, mode='rb') as ifm_file:
    #    # read file returns file data as bytes[]
    #    runway_ifm_file_contents = ifm_file.read()

    #runway_ifm = rpc_client.load_ifm_only(runway_ifm_file_contents, os.path.getsize(runway_ifm_file_name))
    #runway_ofm = rpc_client.allocate_ofm(os.path.getsize(runway_ofm_file_name))

    #model = rpc_client.load_model_only("people.lm")
    #ifm_file_name = "preprocess_002.bin"
    #ofm_file_name = "preprocess_002_ofm.bin"
    #with open(ifm_file_name, mode='rb') as ifm_file:
        # read file returns file data as bytes[]
    #    ifm_file_contents = ifm_file.read()

    #ifm = rpc_client.load_ifm_only(ifm_file_contents, os.path.getsize(ifm_file_name))
    #ifm = rpc_client.allocate_ifm(os.path.getsize(ifm_file_name))
    #ofm = rpc_client.allocate_ofm(os.path.getsize(ofm_file_name))

    #for i in range(100):
    #    start_time = time.perf_counter()
        #ifm = rpc_client.load_ifm_only(ifm_file_contents, os.path.getsize(ifm_file_name))
        #if not ifm:
        #    print("ifm allocation failed")
        #    break
        #ofm = rpc_client.allocate_ofm(os.path.getsize(ofm_file_name))
        #if not ofm:
        #    print(f"ofm allocation failed")
        #    break

    #    rpc_client.upload_ifm(ifm, ifm_file_contents)

        #run_start_time = time.perf_counter()
        #rc, run_time = rpc_client.run_model_phys(model, ifm, ofm)
        #run_end_time = time.perf_counter()

        #print(f"server run time {run_time:.5f}")
        #print(f"{i} - Model runtime {run_end_time - run_start_time:0.4f} seconds")

        #run_start_time = time.perf_counter()
        #rc = rpc_client.run_model_phys(reid_model, reid_ifm, reid_ofm)
        #run_end_time = time.perf_counter()
        #print("reid model done")
        #rpc_client.download_ofm(ofm, "test_ofm.bin", os.path.getsize(ofm_file_name))
        #end_time = time.perf_counter()
        #print(f"{i} - Model runtime {run_end_time - run_start_time:0.4f} seconds")
        #print(f"{i} - Total runtime {end_time - start_time:0.4f} seconds")    
        #rpc_client.free_ifm(ifm)
        #rpc_client.free_ofm(ofm)
        #rc = rpc_client.run_model_phys(runway_model, runway_ifm, runway_ofm)
        #print("runway model done")

    #rpc_client.free_ifm(ifm)
    #rpc_client.free_ofm(ofm)

    #rpc_client.free_model(model)

    #print("connecting to server : port %s" % port)
    #conn = rpyc.connect("192.168.1.12", port)
    #print(conn.root.ping("ping"))

    #remote = conn.root.File()
    #remote.open("test.txt", "wb")
    #local = open("test.txt", "rb")
    #shutil.copyfileobj(local, remote) 

    ### Testing  allocate_array_ifm and upload_array_ifm for CRESTEREO
    rpc_client = Mla_rpc_client("dm7.sjc.sima.ai", port, None)
    import ipdb; ipdb.set_trace(context=30)
    # rpc_client = Mla_rpc_client("localhost", port)
    rpc_client.connect()
    rpc_client.dv_connect()
    # allocate_array_ifm(self, ifm_len, array_size: int=2):
    ifm_size = 691328
    rc, array_mem_ifm = rpc_client.allocate_array_ifm(ifm_size, array_size=2)
    ofm_size = 29491328
    rc, array_mem_ofm = rpc_client.allocate_array_ofm(ofm_size, array_size=1)
    ifm_file_name_array = ["/Users/lam.nguyen/crestereo_demo_arch-tiled_inference/ifm0.npy", "/Users/lam.nguyen/crestereo_demo_arch-tiled_inference/ifm1.npy"]
    ifm_file_contents = []
    for ifm_file in ifm_file_name_array:
        tensor_quant = np.fromfile(ifm_file, dtype=np.int8)
        ifm_file_contents.append(tensor_quant.tobytes())
    rc = rpc_client.upload_array_ifm(array_mem_ifm, ifm_file_contents)
    
    #runway_model = ""
    #rc = rpc_client.run_batch_model_phys(runway_model, 2, array_mem_ifm, ifm_size, array_mem_ofm, ofm_size)
    #rc, ofm_data_array = rpc_client.download_array_ofm(array_mem_ofm, ofm_size)
    #if rc:
    #    print("Get OFM")
    rpc_client.free_array_ifm(array_mem_ifm)
    rpc_client.free_array_ofm(array_mem_ofm)
'''
