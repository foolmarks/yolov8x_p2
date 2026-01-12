from typing import List

import cv2
import gi
import numpy as np
from python_plugin_template import (
    AggregatorTemplate,
    LogLevel,
    SimaaiPythonBuffer,
    logger,
)

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GObject", "2.0")
from gi.repository import GObject, Gst

import utils

# Set the logging level to INFO. Messages with lower severity (e.g., DEBUG) will be ignored.
# Log Levels:
#   EMERG   = 0   # System is unusable (mapped to CRITICAL)
#   ALERT   = 1   # Action must be taken immediately (mapped to CRITICAL)
#   CRIT    = 2   # Critical conditions (mapped to CRITICAL)
#   ERR     = 3   # Error conditions (mapped to ERROR)
#   WARNING = 4   # Warning conditions (mapped to WARNING)
#   NOTICE  = 5   # Normal but significant condition (mapped to INFO)
#   INFO    = 6   # Informational (mapped to INFO)
#   DEBUG   = 7   # Debug-level messages (mapped to DEBUG)
logger.set_level(LogLevel.INFO)

H, W, C = 640, 640, 3


plugin_name = "yolov8xp2_postproc_overlay"  # define PLUGIN_NAME HERE

out_size = int(H * W * 3)  # outsize of plugin in bytes


class MyPlugin(AggregatorTemplate):
    def __init__(self):
        self.out_size = int(640 * 640 * 3)  # outsize of plugin in bytes
        super(MyPlugin, self).__init__(
            plugin_name=plugin_name, out_size=out_size, next_metaparser=False
        )

        self.model_outs = [
            (1, 160, 160, 64),
            (1, 80, 80, 64),
            (1, 40, 40, 64),
            (1, 20, 20, 64),
            (1, 160, 160, 80),
            (1, 80, 80, 80),
            (1, 40, 40, 80),
            (1, 20, 20, 80),
        ]

        self.conf_thres = 0.45
        self.iou_thres = 0.45

    def get_model_outputs(self, input_buffer):
        start = 0
        model_outputs = []
        # first get box inputs
        for out_shape in self.model_outs:
            seg_size = np.prod(out_shape)
            arr = input_buffer[start : start + seg_size].reshape(out_shape)
            model_outputs.append(arr)
            start += seg_size

        return model_outputs

    def run(
        self, input_buffers: List[SimaaiPythonBuffer], output_buffer: bytes
    ) -> None:
        """
        Define your plugin logic HERE
        Inputs:
        input_buffers List[SimaaiPythonBuffer]: List of input buffers
        Object of class SimaaiPythonBuffer has three fields:
        1. metadata MetaStruct Refer to the structure above
        2. data bytes - raw bytes of the incoming buffer
        3. size int - size of incoming buffer in bytes
        """

        # yolov8x-p2 output
        model_buffer = np.frombuffer(input_buffers[0].data, dtype=np.float32, count=-1)

        # original image
        orig_image = np.frombuffer(
            input_buffers[1].data, dtype=np.uint8, count=-1
        ).reshape(H, W, C)

        """
        model_buffer is a numpy array of shape (4896000,) with data type np.float32
        The 8 model outputs have shapes:
            (1, 160, 160, 64)
            (1, 80, 80, 64)
            (1, 40, 40, 64)
            (1, 20, 20, 64)
            (1, 160, 160, 80)
            (1, 80, 80, 80)
            (1, 40, 40, 80)
            (1, 20, 20, 80)
        """

        # list of numpy arrays, shapes as listed above
        model_outputs = self.get_model_outputs(model_buffer)

        # Postprocess in 640x640 space
        boxes_640, scores, class_ids = utils.postprocess_yolov8x_p2_4o(
            model_outputs,
            conf_thr=self.conf_thres,
            iou_thr=self.iou_thres,
            num_classes=80,
            apply_class_sigmoid=True,
        )

        img_bgr = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
        annotated = utils.draw_detections(
            img_bgr, boxes_640, scores, class_ids, utils.COCO_CLASSES
        )

        # write file
        ok = cv2.imwrite("/tmp/output.png", annotated)
        if ok:
            logger.info("Wrote PNG file to /tmp/output.png")

        # output to fakesink
        data = annotated.flatten().tobytes()
        output_buffer[: len(data)] = data


GObject.type_register(MyPlugin)
__gstelementfactory__ = (plugin_name, Gst.Rank.NONE, MyPlugin)
