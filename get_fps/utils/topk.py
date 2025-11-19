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
This is used to calculate the topk, and drawing the bounding box.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class BBOX_T:
    """
    This class of BBOX_T

    :param x1: the original image dimension x.
    :param y1: the original image dimension y.
    :param w:  width of letterbox.
    :param h:  height of letterbox.
    :param score: confident score.
    :param class_id: class_id.
    """
    def __init__(self, x1: int, y1: int, w: int, h: int, score : float, class_id: int):
        self.x1 = x1
        self.y1 = y1
        self.w = w
        self.h = h
        self.score = score
        self.class_id = class_id

def topk(heatmap, offset, inputSize, K, mpk_attrs):
    """
    Get the K largest elements of the given input tensor along a given dimension.

    :param heatmap : a heatmap tensor.
    :param offset : a offset tensor.
    :param inputSize : a inputSize tensor.
    :param K : number requests
    :param mpk_attrs : an MPK attribute
    :return: list of bounding box.
    """

    N, H, W, C = heatmap.shape
    heatmapDepth = mpk_attrs.detessellation_slice_shape[0][2]
    scoreThreshold = (int)(0.4 * 32768)
    origImageHeight = mpk_attrs.original_frame_height
    origImageWidth = mpk_attrs.original_frame_width
    frameHeight = mpk_attrs.detessellation_frame_shape[0][1]
    frameWidth = mpk_attrs.detessellation_frame_shape[0][2]
    frameSize = H * W # 128 , 128
    hmScalar = float(1.0/32768.0)

    # find top scores
    topk = np.argpartition(heatmap.flatten(), -K)[-K:]

    # scaling factor back to original image coordinate, plus letterbox offset
    # convert to the original frame
    upScale = max(float(origImageWidth)/frameWidth, float(origImageHeight)/frameHeight)
    dx = float((frameWidth * upScale - origImageWidth) / 2) # offsize of letter box
    dy = float((frameHeight * upScale - origImageHeight) / 2) # offsize

    #print("Upscaling factor: {}, Letterbox dx = {}, dy {}".format(upScale, dx, dy))
    # output
    k = 0
    outBBoxList = [] # list of bbox_t

    # generate bounding boxes
    while k < K:
        # check score against the threshold
        if (heatmap.flat[topk[k]] < scoreThreshold):
            continue

        # winner location in NHWC - > center .
        # idx = (row * Width + col) * Depth + ch
        n, row, col, ch = np.unravel_index(topk[k], shape=(N, H, W, C))

        # Offset
        offset_sq = np.squeeze(offset, axis=0)
        offset_x = float(offset_sq[row,col,0]) # get the item at index mlcByteOffset
        offset_y = float(offset_sq[row, col,1]) # get the item at index mlcByteOffset + 1

        # InputSize
        inputSize_sq = np.squeeze(inputSize, axis=0)
        width = float(inputSize_sq[row,col,0])
        height = float(inputSize_sq[row,col,1])

        # center
        cx = float(col + offset_x)
        cy = float(row + offset_y)

        # compute top-left corner from (width, height)
        x1 = float(cx - width / 2.0)
        y1 = float(cy - height / 2.0)

        # adjust for the original image dimension and letterbox
        x1 = x1 * upScale - dx
        y1 = y1 * upScale - dy
        width = width * upScale
        height = height * upScale

        # construct a bounding box *****
        bb_x1 = int(x1) if x1 > 0 else 0
        bb_y1 = int(y1) if y1 > 0 else 0
        bb_w = int(width) if ((bb_x1 + width) < origImageWidth) else origImageWidth - bb_x1
        bb_h = int(height) if ((bb_y1 + height) < origImageHeight) else origImageHeight - bb_y1
        bb_score = heatmap.flat[topk[k]] * hmScalar
        bb_class_id = ch;

        # save to outputbuffer
        outBBoxList.append(BBOX_T(bb_x1, bb_y1, bb_w, bb_h, bb_score, bb_class_id))
        k += 1

    return outBBoxList
