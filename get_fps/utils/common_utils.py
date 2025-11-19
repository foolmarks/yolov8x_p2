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
import os.path as osp
import pickle

filexts = ".tar.gz"

def check_image_file_extension(fp):
    """
    Check the image file extension
    :param fp: the fullpath of image.
    :return: bool
    """
    if os.path.splitext(fp)[-1].lower() in [".jpg", ".png", ".jpeg", ".tiff", ".bpm"]:
        return True
    return False


def check_video_file_extension(video_path):
    """
    Check the video file extension
    :param video_path: the fullpath of video.
    :return: bool
    """
    if os.path.splitext(video_path)[-1].lower() in [".mov", ".avi", ".mp4", ".mpeg"]:
        return True
    return False


def check_tar_gz_file(filename):
    """
    Check if file is tar.gz
    :param filename: filename.
    :return: bool
    """
    if filename.endswith(filexts):
        return True
    return False

def get_all_files( rootpath,
                include_type=None,
                include_extns=[],
                exclude_extns=[],
                only_basename=False,
):
    """
    Get all files given in rootpath.
    :param rootpath: rootpath
    :param include_type: 'image'/'video'
    :param include_extns: list of valid extension ['.jpg', '.json']
    :param exclude_extns: list of valid extension ['.jpg', '.json', '.MD']
    :param only_basename: only get the basename of the path
    """
    all_files = []
    for root, dir, files in os.walk(rootpath):
        root_split_list = root.split("/")
        for f in files:
            basename_no_ext, extn = osp.splitext(f)
            if extn in exclude_extns:
                continue
            if include_extns and extn not in include_extns:
                continue
            if include_type == 'image':
                if not check_image_file_extension(f):
                    continue
            elif include_type == 'video':
                if not check_video_file_extension(f):
                    continue
            full_p = osp.join(root, f)

            if osp.isfile(full_p):
                if only_basename:
                    all_files.append(osp.join(root, basename_no_ext))
                else:
                    all_files.append(full_p)

    return all_files
