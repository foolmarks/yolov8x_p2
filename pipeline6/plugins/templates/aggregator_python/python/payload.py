import numpy as np 
import cv2
from python_plugin_template import AggregatorTemplate, logger, LogLevel
from python_plugin_template import SimaaiPythonBuffer, MetaStruct
import gi
from typing import List, Tuple
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GObject', '2.0')
from gi.repository import Gst, GObject, GstBase

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
"""
To use Metadata fieds from the input buffers:  
Parse the MetaStruct object. It has the following 4 fields:  
class MetaStruct:
    def __init__(self, buffer_name, stream_id, timestamp, frame_id):
        self.buffer_name = buffer_name
        self.stream_id = stream_id
        self.timestamp = timestamp
        self.frame_id = frame_id

Note: While using simaai plugins downstream of this plugin, make sure that next_metaparser is set to True in the constructor of the plugin.
"""

plugin_name = "<PLUGIN_NAME>"   #define PLUGIN_NAME HERE
out_size = int(1280 * 720 * 1.5)  # outsize of plugin in bytes
class MyPlugin(AggregatorTemplate):
    def __init__(self):
        print(f"Out Size for {plugin_name}: {out_size}")
        super(MyPlugin, self).__init__(plugin_name=plugin_name, out_size=out_size, next_metaparser=False)
    
    def run(self, input_buffers: List[SimaaiPythonBuffer], output_buffer: bytes) -> None:
        """
        Define your plugin logic HERE
        Inputs:
        input_buffers List[SimaaiPythonBuffer]: List of input buffers  
        Object of class SimaaiPythonBuffer has three fields:  
        1. metadata MetaStruct Refer to the structure above
        2. data bytes - raw bytes of the incoming buffer  
        3. size int - size of incoming buffer in bytes
        4. array List[np.ndarray]: List of model output tensors if your plugin follows a detess_dequant plugin, empty list otherwise.
        """
        pass

GObject.type_register(MyPlugin)
__gstelementfactory__ = (plugin_name, Gst.Rank.NONE, MyPlugin)