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
This is used to create MPK Attrs for models that run exclusively
on the MLA, and for which we just want to obtain FPS KPIs.
"""
from dataclasses import dataclass, field
from typing import (
    List, Tuple
)
import numpy as np

@dataclass
class MPKAttrs(object):
    """
    This class includes information from the MPK JSON that is
    needed to run an .lm file on a DevKit and obtain FPS KPIs.

    :param tessellation_slice_height: slice_height for tessellation.
    :param tessellation_slice_width: slice_width for tessellation.
    :param tessellation_slice_channels: slice_channels for tessellation.
    :param tessellation_align_c16: False when interfacing to MLA; True to force 16-channel alignment for testing.
    :param ifm_size : Expected the size of ifm (nbytes)
    :param ofm_size : Expected the size of ofm (nbytes)
    """
               
    # Tessellation Transform
    tessellation_slice_height: int = field(default_factory=int)
    tessellation_slice_width: int = field(default_factory=int)
    tessellation_slice_channels: int = field(default_factory=int)
    tessellation_align_c16: bool = field(default_factory=bool)

    ifm_size: int = field(default_factory=int)
    ofm_size: int = field(default_factory=int)
    batch_size: int = field(default_factory=int)

    def __init__(self, mpk_dict):
        # Skip over plugins until we get to the tessellation plugin.
        num_plugins = len(mpk_dict['plugins'])
        for index in range(num_plugins):
            plugin = mpk_dict['plugins'][index]['config_params']
            if 'kernel' in plugin and plugin['kernel'] == "tessellation_transform":
                break

        assert plugin['kernel'] == "tessellation_transform", \
            "Tessellation plugin expected in MPK JSON file."

        # The plugin must be a tessellation plugin.
        plugin = mpk_dict['plugins'][index]['config_params']
        assert plugin['kernel'] == "tessellation_transform", \
            "Tessellation plugin expected in MPK JSON file."
        self.tessellation_slice_height = plugin['params']['slice_shape'][0]
        self.tessellation_slice_width = plugin['params']['slice_shape'][1]
        self.tessellation_slice_channels = plugin['params']['slice_shape'][2]
        self.tessellation_align_c16 = plugin['params']['align_c16']
        self.ifm_size = mpk_dict['plugins'][index]['output_nodes'][0]['size']

        # The following plugin must be an MLA plugin.
        plugin = mpk_dict['plugins'][index + 1]
        assert plugin['processor'] == "MLA", \
            "MLA plugin expected in MPK JSON file."
        self.ofm_size = plugin['output_nodes'][0]['size']
        if 'actual_batch_size' in plugin['config_params']:
            self.batch_size = plugin['config_params']['actual_batch_size']
        else:
            self.batch_size = 1
