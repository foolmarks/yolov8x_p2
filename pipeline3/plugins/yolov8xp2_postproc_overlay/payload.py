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

H, W, C = 720, 1280, 3


plugin_name = "yolov8xp2_postproc_overlay"  # define PLUGIN_NAME HERE

out_size = int(H * W * C)  # outsize of plugin in bytes


class MyPlugin(AggregatorTemplate):
    def __init__(self):
        self.out_size = int(1280 * 720 * 3)  # outsize of plugin in bytes
        super(MyPlugin, self).__init__(
            plugin_name=plugin_name, out_size=out_size, next_metaparser=False
        )

    def nv12_to_bgr(self, nv12: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Convert an NV12 (Y + interleaved UV) image to BGR using OpenCV.

        Parameters
        ----------
        nv12:
            NV12 buffer as a NumPy uint8 array. Common shapes are:
              - (height * 3 // 2, width)  (recommended)
              - (height * width * 3 // 2,) flat buffer
        width, height:
            Image dimensions.

        Returns
        -------
        bgr:
            (height, width, 3) uint8 BGR image.
        """

        if nv12.dtype != np.uint8:
            raise TypeError(f"Expected nv12 dtype=uint8; got {nv12.dtype}")

        expected = height * width * 3 // 2

        if nv12.ndim == 1:
            if nv12.size != expected:
                raise ValueError(
                    f"Expected flat NV12 of {expected} bytes; got {nv12.size}"
                )
            yuv = nv12.reshape((height * 3 // 2, width))

        elif nv12.ndim == 2:
            if nv12.shape != (height * 3 // 2, width):
                raise ValueError(
                    f"Expected NV12 shape {(height * 3 // 2, width)}; got {nv12.shape}"
                )
            yuv = nv12

        else:
            raise ValueError(f"Expected NV12 as 1D or 2D array; got ndim={nv12.ndim}")

        # Ensure contiguous for cvtColor
        yuv = np.ascontiguousarray(yuv)

        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        return bgr

    def run(
        self, input_buffers: List[SimaaiPythonBuffer], output_buffer: bytes
    ) -> None:
        logger.info("STARTED")

        image = np.frombuffer(input_buffers[0].data, dtype=np.uint8, count=-1).reshape(
            H * 3 // 2, W
        )

        image = self.nv12_to_bgr(image, width=W, height=H)

        # write file
        ok = cv2.imwrite("/tmp/output.png", image)
        if ok:
            logger.info("Wrote PNG file to /tmp/output.png")

        # output to fakesink
        data = image.flatten().tobytes()
        output_buffer[: len(data)] = data


GObject.type_register(MyPlugin)
__gstelementfactory__ = (plugin_name, Gst.Rank.NONE, MyPlugin)
