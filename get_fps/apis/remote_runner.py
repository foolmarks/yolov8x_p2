#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************
# Code owner: Vimal Nakum
###########################################################################
"""
Runs a model remotely on the board, with full PePP pipeline on MLSoC.
"""
import os
import argparse
from os import path
from typing import List, Dict
import time
import json
import warnings
import sys
import os
import logging
import getpass
import json
import subprocess
import webbrowser

PROJECT_ROOT = os.path.abspath(sys.path[0].rsplit('/',1)[0])
sys.path.append(PROJECT_ROOT)
DEFAULT_BATCH_SIZE = 1

#suppress warnings
warnings.filterwarnings('ignore')

from mla_rt_client import Mla_rpc_client as mla_rt_client

DEVKIT_FOLDER="/home/sima"
DEFAULT_PORT=8000
DEFAULT_USERNAME='sima'
DEFAULT_VIRTUAL_ENV='sima_env'
DEFAULT_SIMA_TEMP_FOLDER = "sima_temp"
DEFAULT_SIMA_REQ_TXT = DEFAULT_SIMA_TEMP_FOLDER + "/requirements.txt"

def handle_interrupt(signal, frame):
    print("Ctrl+C detected. Exiting...")
    # Perform cleanup or other actions here
    # ...
    # Exit the program
    sys.exit(-1)


def start_gst_launch(gst_port):
    """
    Launching on host
    :param gst_port : port.
    """

    print('Starting to gst_launch on host')
    gst_launch_cmd = ["gst-launch-1.0", f"udpsrc port={gst_port}"," ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! 'video/x-h264,stream-format=byte-stream,alignment=au' ! avdec_h264 ! fpsdisplaysink"]
    gst_launch_cmd = ' '.join(gst_launch_cmd)
    gst_launch_connection = subprocess.Popen(gst_launch_cmd, stdout=subprocess.PIPE, shell=True)
    print(f"Launching on host with {gst_launch_connection.pid}")

# create_forward_tunnel and scp_files_to_remote are copied from ssh_utils to make this more self contained
def create_forward_tunnel(args, password: str, max_attempts: int):
    """
    Create the Forwarding from host

    :param args: ArgumentParser.
    :param password: password.
    :param max_attempts: max_attempts.
    """
    local_port = args.dv_port
    ssh_connection = None
    print("Creating the Forwarding from host")
    for attempt in range(max_attempts):
        if password == '':
            cmd = ["ssh", "-f", "-N", "-L", f"{local_port}:localhost:{args.dv_port} {args.dv_user}@{args.dv_host}"]
        else:
            cmd = ["sshpass", "-p", "{}".format(password), "ssh", "-f", "-N", "-L", f"{local_port}:localhost:{args.dv_port} {args.dv_user}@{args.dv_host}"]
        cmd = ' '.join(cmd)
        logger.debug(f'Attempt {attempt} to forward local port {local_port} to {args.dv_port} with cmd {cmd}')
        connection = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        rc = connection.wait()
        if rc != 0:
            logger.debug(f'Attempt {attempt} failed to forward local port {local_port} to {args.dv_port}')
            local_port += 1
            continue
        else:
            logger.debug(f'Attempt {attempt} successful to forward local port {local_port} to {args.dv_port} with pid {connection.pid}')
            ssh_connection = connection
            break
    return ssh_connection, local_port


def scp_files_to_remote(args, file_path, password: str, dest: str, max_attempts: int):
    """
    SCP the files to Remote

    :param args: ArgumentParser.
    :param dest: target directory.
    :param max_attempts: max_attempts.
    """
    ## Check the exist file and directory
    assert os.path.isfile(file_path), f"[Error]: {file_path} doesn't exist"
    scp_file = None
    ## Check OpenSSH version (need -O flag in v9.0 as scp is deprecated in that version)
    output = subprocess.check_output(['ssh', '-V'], stderr=subprocess.STDOUT)
    version_line = output.decode().strip().splitlines()[0]
    version_string = version_line.split(' ')[0]
    ## If Mac OS is Ventura (version:13.2.1), and Openssh is OpenSSH_9.0p1 or later. Need to add -O due to https://www.openssh.com/txt/release-9.0.
    scp_string = f"scp{' -O' if '9.0' in version_string else ''}"
    for attempt in range(max_attempts):
        if password == '':
            scp_cmd = [scp_string, f"{file_path}", f"{args.dv_user}@{args.dv_host}:/{DEVKIT_FOLDER}"]
        else:
            scp_cmd = ["sshpass", "-p", "{}".format(password), scp_string, f"{file_path}", f"{args.dv_user}@{args.dv_host}:/{DEVKIT_FOLDER}"]
        scp_cmd = ' '.join(scp_cmd)
        scp_connection = subprocess.Popen(scp_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = scp_connection.communicate()
        if scp_connection.returncode != 0:
            logger.debug(f'Attempt {attempt} failed to scp {file_path} to {DEVKIT_FOLDER}')
            logger.debug(f'ERROR {error}')
            continue
        else:
            logger.debug(f'Attempt {attempt} successful to scp {file_path} to {DEVKIT_FOLDER} with pid {scp_connection.pid}')
            scp_file = scp_connection
            break
    return scp_file

# modules to remove from requirements.txt as they are already installed on the board
skip_module_list = ['sima','opencv_python', 'torch', 'torchvision']

def get_requirements_txt(model_file_path:str) -> str:
    # untar the file in the local directory and run pipreqs to create requirements.txt
    cmd_str = "mkdir -p " + DEFAULT_SIMA_TEMP_FOLDER
    result = subprocess.run(cmd_str, shell=True)
    if result.returncode != 0:
        logger.error("mkdir failed")
        return None
    
    cmd_str = "tar -xzf " + args.model_file_path + " -C " + DEFAULT_SIMA_TEMP_FOLDER
    result = subprocess.run(cmd_str, shell=True)
    if result.returncode != 0:
        logger.error("untar failed")
        return None

    cmd_str = "pipreqs --force " + DEFAULT_SIMA_TEMP_FOLDER
    result = subprocess.run(cmd_str, shell=True)
    if result.returncode != 0:
        logger.error("untar failed")
        return None
    
    # check if requirements.txt is created
    if os.path.exists(DEFAULT_SIMA_REQ_TXT):        
        # remove duplicate modules from the file and
        # remove sima, opencv, numpy from the list as they are already
        # installed on the board        
        with open(DEFAULT_SIMA_REQ_TXT, 'r') as in_file, open(DEFAULT_SIMA_TEMP_FOLDER + "/requirements1.txt", 'w') as out_file:
            output_set = set()
            for input_line in in_file:
                input_line_stripped = input_line.strip()
                parts = input_line_stripped.split("==")
                if parts[0] in skip_module_list:
                    continue
                # just keep the module name and if it is not seen, write that line
                # to the output file
                if parts[0] not in output_set:
                    out_file.write(input_line)
                    output_set.add(parts[0])

        cmd_str = "mv " + DEFAULT_SIMA_TEMP_FOLDER + "/requirements1.txt " + DEFAULT_SIMA_REQ_TXT
        result = subprocess.run(cmd_str, shell=True)
        if result.returncode != 0:
            logger.error("mv failed")
            return None
                
        return DEFAULT_SIMA_REQ_TXT
    return None

          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remotely run models on MLSoC')
    parser.add_argument('--dv_host', required=True, type=str, help='DevKit IP Address / FQDN')
    parser.add_argument('--dv_port', required=False, type=int, default=DEFAULT_PORT, help='DevKit port on which the mla_rt_service is running')
    parser.add_argument('--dv_user', required=False, type=str, default=DEFAULT_USERNAME, help='DevKit ssh username')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument('--bypass_tunnel', required=False, action='store_true', help= 'set to bypass ssh tunnel')
    parser.add_argument('--model_command', nargs='+', required=True, type=str, help='Test file to run remotely')
    parser.add_argument('--model_file_path', required=True, type=str, help='Model tar.gz file including lm and jsons')
    parser.add_argument('--run_time', required=False, type=int, default=60, help='Model run time in seconds, 0 is invalid')
    parser.add_argument('--gst_port', required=False, type=int, help='Port for gst_lauch')
    args = parser.parse_args()

    # Create the logfile with BasicConfig
    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.WARN
        
    logger = logging.getLogger('remote_runner_logger')
    logger.setLevel(logging.DEBUG)
    
    # Create a file handler and set the level to INFO
    file_handler = logging.FileHandler('remote_runner.log')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    # Create a console handler and set the level to ERROR, this would print on the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    logger.addHandler(console_handler)

    logger.debug('This message should go to the log file')

    password = ''
    max_attempts = 10
    if not args.bypass_tunnel:
        ssh_connection, local_port = create_forward_tunnel(args, password, max_attempts)
        
        if ssh_connection is None:
            logger.error(f'Failed to connect to {args.dv_host} after {max_attempts} attempts')
            sys.exit(-1)

        # we start to work with the local_port from now on
        args.dv_port = local_port

    model_reqs_txt = get_requirements_txt(args.model_file_path)
    if not model_reqs_txt:
        logger.error(f'Unable to generate requirements.txt for model tar.gz file')
        sys.exit(-1)    
        
    # Copy the .lm or .tar.gz model file to the board.
    print(f"Copying Model files to the DevKit")
    scp_file = scp_files_to_remote(args, args.model_file_path, password, DEVKIT_FOLDER, max_attempts)
    if scp_file is None:
        logger.error(f'Failed to scp the model file after {max_attempts}')
        sys.exit(-1)    

    print(f"Copying Model Requirements to the DevKit")
    scp_file = scp_files_to_remote(args, model_reqs_txt, password, DEVKIT_FOLDER, max_attempts)
    if scp_file is None:
        logger.error(f'Failed to copy requirements txt {max_attempts}')
        sys.exit(-1)    
 
    # get rpc client to run model remotely
    rpc_client = mla_rt_client(args.dv_host, args.dv_port, False)
    if rpc_client.connect() is False:
        raise RuntimeError("ERROR: RPYC connect failed")
   
    # create virtual env
    print(f"Creating virtual environment on the DevKit")
    rc, result = rpc_client.create_virtual_env(DEFAULT_VIRTUAL_ENV)
    if rc != 0:
        # Print error outputs
        logger.error(f"create virtual env failed rc {rc} result rc {result.returncode} stdout {result.stdout}, stderr {result.stderr}")
        raise RuntimeError(f"ERROR: create virtual env failed - rc {rc} result {result }")
    else:
        print('Successfully created virtual env')

    if args.gst_port:
        start_gst_launch(args.gst_port)
    
    if not args.run_time:
        print(f"run_time is 0, overriding it to 60 seconds")
        timeout = 60
    else:
        timeout = args.run_time

    # run the model remotely
    command = args.model_command
    
    # check and remove if user specified python in the command name
    command_words = command[0].split()
    if "python" == command_words[0] or "python3" == command_words[0]:
        command_words = command_words[1:]
        command = [" ".join(command_words)]

    # check if user specified --run_args otherwise add it after script name
    if "--run_args" not in command[0]:
        command_words = command[0].split()
        add_run_args = [command_words[0], "--run_args"] + command_words[1:]
        command = [" ".join(add_run_args)]
        
    print(f"Running Model on the DevKit")
    rc, output, error = rpc_client.run_remote_model(command, os.path.basename(args.model_file_path), timeout, DEFAULT_VIRTUAL_ENV)
    
    # Remove the 'b' prefix and outer single quotes
    # Replace the escape sequences with their corresponding characters
    cleaned_string = output[2:-1]
    decoded_output = cleaned_string.encode().decode('unicode_escape')
    print(f"stdout: {decoded_output}")
    
    # Since the model command is run for a time duration, it always gets "Command timed out" error
    # so supressing that error
    if "Command timed out" not in error:
        print(f"returncode: {rc}")
        print(f"stderr: {error}")

    # remove virtual env
    if rpc_client.delete_virtual_env(DEFAULT_VIRTUAL_ENV) is False:
        raise RuntimeError("ERROR: delete virtual env failed")
    
    if not args.bypass_tunnel:
        ssh_connection.kill()


