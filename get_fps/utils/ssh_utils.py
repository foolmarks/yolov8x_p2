#########################################################
# Copyright (C) 2022 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
# Code owner: Lam Nguyen
#########################################################
"""
This includes all common functions.
"""

import os
import logging
import argparse
import subprocess
import platform
LM_FOLDER="/home/sima"

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
        logging.debug(f'Attempt {attempt} to forward local port {local_port} to {args.dv_port} with cmd {cmd}')
        connection = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        rc = connection.wait()
        if rc != 0:
            logging.debug(f'Attempt {attempt} failed to forward local port {local_port} to {args.dv_port}')
            local_port += 1
            continue
        else:
            logging.debug(f'Attempt {attempt} successful to forward local port {local_port} to {args.dv_port} with pid {connection.pid}')
            ssh_connection = connection
            break
    return ssh_connection, local_port


def create_forward_tunnel_generic(dv_port, dv_user, dv_host, password: str, max_attempts: int):
    """
    Create the Forwarding from host with port, user and other parameters, rather than using it
    from args
    create_forward_tunnel is used in many places, so not easy to modify. Currently, this is 
    used for creating a tunnel for zmq port.
    """
    local_port = dv_port
    ssh_connection = None
    print("Creating the Forwarding from host")
    for attempt in range(max_attempts):
        if password == '':
            cmd = ["ssh", "-f", "-N", "-L", f"{local_port}:localhost:{dv_port} {dv_user}@{dv_host}"]
        else:
            cmd = ["sshpass", "-p", "{}".format(password), "ssh", "-f", "-N", "-L", f"{local_port}:localhost:{dv_port} {dv_user}@{dv_host}"]
        cmd = ' '.join(cmd)
        logging.debug(f'Attempt {attempt} to forward local port {local_port} to {dv_port} with cmd {cmd}')
        connection = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        rc = connection.wait()
        if rc != 0:
            logging.debug(f'Attempt {attempt} failed to forward local port {local_port} to {dv_port}')
            local_port += 1
            continue
        else:
            logging.debug(f'Attempt {attempt} successful to forward local port {local_port} to {dv_port} with pid {connection.pid}')
            ssh_connection = connection
            break
    return ssh_connection, local_port


def scp_files_to_remote(args, file_path, password: str, dest: str, max_attempts: int, dv_user: str = '', dv_host: str = ''):
    """
    SCP the files to Remote

    :param args: ArgumentParser.
    :param dest: target directory.
    :param max_attempts: max_attempts.
    """
    #lm_file_path = args.lm_file_path

    devkit_username = ''
    devkit_hostname = ''
    if args == None:
        assert dv_user != '' and dv_host != '', "Please provide username and hostname"
        devkit_username = dv_user
        devkit_hostname = dv_host
    else:
        devkit_username = args.dv_user
        devkit_hostname = args.dv_host

    print("Copying the model files to DevKit")
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
            scp_cmd = [scp_string, f"{file_path}", f"{devkit_username}@{devkit_hostname}:/{LM_FOLDER}"]
        else:
            scp_cmd = ["sshpass", "-p", "{}".format(password), scp_string, f"{file_path}", f"{devkit_username}@{devkit_hostname}:/{LM_FOLDER}"]
        scp_cmd = ' '.join(scp_cmd)
        scp_connection = subprocess.Popen(scp_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = scp_connection.communicate()
        if scp_connection.returncode != 0:
            logging.debug(f'Attempt {attempt} failed to scp {file_path} to {LM_FOLDER}')
            logging.debug(f'ERROR {error}')
            continue
        else:
            logging.debug(f'Attempt {attempt} successful to scp {file_path} to {LM_FOLDER} with pid {scp_connection.pid}')
            scp_file = scp_connection
            break
    return scp_file

def get_layer_stats_yaml_file(args, password: str, remote_file_path: str, max_attempts: int):
    # layer processing ads _output to the file name
    layer_output_file_name = os.path.splitext(os.path.basename(remote_file_path))[0] + "_output.yaml"
    layer_output_file_path = "/home/sima/" + layer_output_file_name
    scp_files_from_remote(args, password, layer_output_file_path, max_attempts)

    yaml_remove_strings = ['start_cycle', 'end_cycle', 'layer_latency', 'active_time', 'l2_freeze', 'iq_freeze']    
    
    filter_lines = []
    with open(layer_output_file_name, 'r') as yaml_file:
        lines = yaml_file.read().splitlines()
        
    for line in lines:
        if not any (string in line for string in yaml_remove_strings):
            filter_lines.append(line)
                
    with open(layer_output_file_name, 'w') as yaml_file:
        yaml_file.writelines('\n'.join(filter_lines))
    
def scp_files_from_remote(args, password: str, remote_file_path: str, max_attempts: int):
    """
    SCP a remote file into current directory

    :param args: ArgumentParser.
    :param remote_file_path: remote directory.
    :param max_attempts: max_attempts.
    """
    if not remote_file_path:
        return

    scp_file = None
    ## Check OpenSSH version (need -O flag in v9.0 as scp is deprecated in that version)
    output = subprocess.check_output(['ssh', '-V'], stderr=subprocess.STDOUT)
    version_line = output.decode().strip().splitlines()[0]
    version_string = version_line.split(' ')[0]
    ## If Mac OS is Ventura (version:13.2.1), and Openssh is OpenSSH_9.0p1 or later. Need to add -O due to https://www.openssh.com/txt/release-9.0.
    scp_string = f"scp{' -O' if '9.0' in version_string else ''}"
    for attempt in range(max_attempts):
        if password == '':
            scp_cmd = [scp_string, f"{args.dv_user}@{args.dv_host}:{remote_file_path}", "."]
        else:
            scp_cmd = ["sshpass", "-p", "{}".format(password), scp_string, f"{args.dv_user}@{args.dv_host}:{remote_file_path}", "."]
        scp_cmd = ' '.join(scp_cmd)
        scp_connection = subprocess.Popen(scp_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = scp_connection.communicate()
        if scp_connection.returncode != 0:
            logging.debug(f'Attempt {attempt} failed to scp {remote_file_path}')
            logging.debug(f'ERROR {error}')
            continue
        else:
            logging.debug(f'Attempt {attempt} successful to scp {remote_file_path} with pid {scp_connection.pid}')
            scp_file = scp_connection
            break

def run_remote_command(args: str, command: str, password: str, max_attempts: int, command_list=None):
    """
    Runs remote command on the host using SSH

    :param args: ArgumentParser.
    :param command: remote command to run as a string
    :param password: user password from command line
    :param max_attempts: max_attempts.
    """
    remote_command = ["ssh", f"{args.dv_user}@{args.dv_host}", f"{command}"]
    for attempt in range(max_attempts):
        if password != '':
            remote_command = ["sshpass", "-p", "{}".format(password), "ssh", f"{args.dv_user}@{args.dv_host}", f"{command}"]
        remote_command = ' '.join(remote_command)
        scp_connection = subprocess.Popen(remote_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = scp_connection.communicate()
        # Decode the stdout bytes to a string
        stdout_str = output.decode('utf-8')
        error_str = error.decode('utf-8')
        if scp_connection.returncode != 0:
            logging.debug(f'Attempt {attempt} failed to run command {command}')
            logging.debug(f'ERROR {error}')
            continue
        else:
            logging.debug(f'Attempt {attempt} successful to run command {command}')
            break
    return stdout_str, error_str
