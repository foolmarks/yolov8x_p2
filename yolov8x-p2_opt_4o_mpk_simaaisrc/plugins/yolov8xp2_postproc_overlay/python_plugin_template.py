import gi
import os
import sys
import json
import logging
import threading
from enum import Enum
import numpy as np 
from typing import List, Dict, Optional, Tuple
import cv2
from dataclasses import dataclass
import struct
import traceback
import ctypes
from mpk_parser import Parser, sima_mpk

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GObject', '2.0')
from gi.repository import Gst, GObject, GstBase
from enum import Enum
import time

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
APP_BASE_PATH = os.path.join(CURR_DIR, "../..")

MANIFEST_JSON_PATH = os.path.join(APP_BASE_PATH, "manifest.json")
PROCESS_MLA_RESOURCES = os.path.join(APP_BASE_PATH, "share/process_mla")


Gst.init(None)
class MetadataStruct(ctypes.Structure):
    _fields_ = [
        ("buffer_id", ctypes.c_int64),
        ("frame_id", ctypes.c_int64),
        ("timestamp", ctypes.c_uint64),
        ("buffer_offset", ctypes.c_int64),
        ("pcie_buffer_id", ctypes.c_int64),
        ("stream_id_len", ctypes.c_uint32),  # Length of stream_id string
        ("buffer_name_len", ctypes.c_uint32), # Length of buffer_name string
    ]

SIMAAI_META_STR = "GstSimaMeta"
PLUGIN_CPU_TYPE = "APU"

class LogLevel(Enum):
    EMERG = 0    # System is unusable (mapped to CRITICAL)
    ALERT = 1    # Action must be taken immediately (mapped to CRITICAL)
    CRIT = 2     # Critical conditions (mapped to CRITICAL)
    ERR = 3      # Error conditions (mapped to ERROR)
    WARNING = 4  # Warning conditions (mapped to WARNING)
    NOTICE = 5   # Normal but significant condition (mapped to INFO)
    INFO = 6     # Informational (mapped to INFO)
    DEBUG = 7    # Debug-level messages (mapped to DEBUG)

class Logger:
    _instance = None
    _lock = threading.Lock()  # Ensure thread safety for singleton creation

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Logger, cls).__new__(cls)
                cls._instance._initialized = False  # Mark as not yet initialized
        return cls._instance

    def __init__(self, log_file="/var/log/simaai.log", enable_console=False):
        if self._initialized:
            return  # Skip reinitialization

        self.enable_console = enable_console
        self.logger = logging.getLogger("SimaAI")
        self.logger.setLevel(LogLevel.INFO.value)  # Default level
        self.logger.propagate = False  # Disable propagation to parent loggers
        self.level = LogLevel.INFO.value

        # Remove all existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] %(message)s')

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if self.enable_console:
            # Stream handler (for console)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(self.level)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

        sys.excepthook = self.log_exception

        self.log_methods = {
            LogLevel.EMERG: self.logger.critical,
            LogLevel.ALERT: self.logger.critical,
            LogLevel.CRIT: self.logger.critical,
            LogLevel.ERR: self.logger.error,
            LogLevel.WARNING: self.logger.warning,
            LogLevel.NOTICE: self.logger.info,
            LogLevel.INFO: self.logger.info,
            LogLevel.DEBUG: self.logger.debug,
        }
        for level in LogLevel:
            setattr(self, level.name.lower(), self._create_log_method(level))
        # Redirect stdout and stderr to logger
        self._redirect_print_to_logger()

        self._initialized = True  # Mark as initialized

    def _create_log_method(self, level: LogLevel):
        """Create a log method for the given level."""
        def log_method(message, *args, **kwargs):
            if level.value <= self.level:  # Check against current log level
                log_func = self.log_methods.get(level, self.logger.info)
                log_func(message, *args, **kwargs)
        return log_method
    
    def _redirect_print_to_logger(self):
        """Redirect print statements to the logger."""
        class StreamToLogger:
            """A file-like stream object to redirect stdout and stderr to a logger."""
            def __init__(self, log_function):
                self.log_function = log_function

            def write(self, message):
                if message.strip():  # Ignore empty messages
                    self.log_function(message.strip())

            def flush(self):
                """Flush method for compatibility."""
                pass

        sys.stdout = StreamToLogger(self.logger.info)
        sys.stderr = StreamToLogger(self.logger.error)

    def set_level(self, level: LogLevel):
        """Set log level using LogLevel enum."""
        self.logger.setLevel(level.value)
        self.level = level.value
        for handler in self.logger.handlers:
            handler.setLevel(level.value)

        # Recreate log methods to respect the new log level
        for log_level in LogLevel:
            setattr(self, log_level.name.lower(), self._create_log_method(log_level))

    def log(self, level: LogLevel, message: str, *args, **kwargs):
        """Log a message at the given level."""
        log_method = self.log_methods.get(level, self.logger.info)
        if level.value > self.level:
            return
        log_method(message, *args, **kwargs)

    def log_exception(self, exc_type, exc_value, exc_traceback):
        """Log uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Call the default excepthook for KeyboardInterrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self.logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
# Read manifest json to read appname to locate log file
current_dir = os.path.dirname(os.path.abspath(__file__))
manifest_config = None
if os.path.isfile(MANIFEST_JSON_PATH):
    manifest_path = os.path.join(current_dir, MANIFEST_JSON_PATH)
    with open(manifest_path) as f:
        manifest_config = json.load(f)
    appId = manifest_config["applications"][0]["appId"]
    app_logfile = f"/tmp/simaai/{appId}_Pipeline/gst_app.log"
else:
    app_logfile = f"/var/log/simaai.log" # use this to test stand alone
logger = Logger(log_file=app_logfile)
class MetaStruct:
    def __init__(self, buffer_name, stream_id, timestamp, frame_id):
        self.buffer_name = buffer_name
        self.stream_id = stream_id
        self.timestamp = timestamp
        self.frame_id = frame_id


class SimaaiPythonBuffer:

    def __init__(self, metadata: MetaStruct, map: Gst.MapInfo, reshape_tensors: Optional [List[np.ndarray]] = None):
        self.metadata = metadata
        self.data = map.data
        self.size = map.size
        self.reshape_tensors = reshape_tensors
        self.array = self._get_tensor_array() if self.reshape_tensors else []
    
    def _get_tensor_array(self):

        fp32_buffer = np.frombuffer(self.data, dtype=np.float32)
        out_list = []
        start = 0
        num_tensors = len(self.reshape_tensors)
        for idx in range(num_tensors):
            tensor_shape = self.reshape_tensors[idx][0]
            offset = int(np.prod(tensor_shape))
            out_list.append(fp32_buffer[start:start+offset])
            start+=offset
        return out_list


class ValueType(Enum):
    INT64 = GObject.TYPE_INT64
    UINT64 = GObject.TYPE_UINT64
    STRING = GObject.TYPE_STRING
    DOUBLE = GObject.TYPE_DOUBLE


def get_monotonic_timestamp():
    # Get the current timestamp in microseconds
    return int(time.monotonic() * 1_000_000)

class AggregatorTemplate(GstBase.Aggregator):
    """
    A Python based gstreamer plugin template. Enables the user to:  
    - Accept incoming buffers from dynamic pads  
    - Define any custom plugin runtime logic  
    User has to only override the run() function
    """
    transmit = GObject.Property(type=bool, default=False, nick="Flag to enable/disable KPI transmission")
    silent = GObject.Property(type=bool, default=False, nick="Flag to enable/disable silent mode")
    config = GObject.Property(type=str, default="some_path", nick="config json path")
    
    __gstmetadata__ = ('AggregatorTemplate', 'Aggregator', 'Custom Python Aggregator', 'YourName')

    __gsttemplates__ = (
        Gst.PadTemplate.new_with_gtype(
            "sink_%u",
            Gst.PadDirection.SINK,
            Gst.PadPresence.REQUEST,
            Gst.Caps.new_any(),
            GstBase.AggregatorPad.__gtype__,
        ),
        Gst.PadTemplate.new_with_gtype(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.new_any(),
            GstBase.AggregatorPad.__gtype__,
        )
    )
    
    def __init__(self, plugin_name, out_size, next_metaparser=False):
        super(AggregatorTemplate, self).__init__()
        self.plugin_name = plugin_name
        self.dynamic_pads = []
        self.src_caps_set = False
        self.timestamp = 0
        self.frame_id = 0
        self.is_pcie = False
        self.in_pcie_buf_id = 0
        self.stream_id = "unknown-stream"
        self.buffer_name = "default"
        self.buffer_id = 0
        self.plugin_id = "python-agg-template"
        self.t0 = None
        self.t1 = None
        self.manifest_json = manifest_config
        self.next_plugin_is_metaparser = next_metaparser
        out_size_user = out_size
        self.mpk_path = None
        self.model_output_shapes = []
        self.grandparent_config = None # config path to the mla config json if pygast plugin is used after detess_dequant
        self.detess_dequant_pad_name = "" # sink pad name that is connected from detess-dequant to PyGast plugin
        # Calculate output buffer size based on metaparser flag
        if self.next_plugin_is_metaparser:
            metadata_size = ctypes.sizeof(MetadataStruct)
            max_string_size = 256  # Maximum size for strings stream_id & buffer_name
            total_size = metadata_size + (2 * max_string_size) + out_size_user
            logger.info(f"Metadata size that needs to be attached as header {metadata_size} bytes")
            logger.info(f"Actual output buffer size from pygast plugin {out_size_user} bytes")
            logger.info(f"Allocating output buffer size for metadata_size + output buffer size from user := {total_size} bytes")
            self.out_size = total_size
        else:
            # Use the user supplied pygast plugin output size when metaparser is disabled
            self.out_size = out_size_user
            logger.info(f"Using direct output buffer size: {self.out_size} bytes")
        self.frame_id = 0
        self.pcie_buffer_id = 0
        # Register the custom metadata during initialization
        self.register_metadata()
        self.metadata_add_failed = False
        self.metadata_len = 0

        #trial. Can you get information about the parent plugin in the init function. 


    def register_metadata(self) -> None:
        try:
            # First check if metadata type already exists
            existing_type = GObject.type_from_name(SIMAAI_META_STR)
            if existing_type != 0:
                print(f"Metadata {SIMAAI_META_STR} is already registered")
                return

            # Register custom metadata using the simplified API
            # Parameters:
            # - metadata name: unique identifier for our metadata
            # - metadata description: empty string as it's optional
            # - init_func: None to use default initialization
            # - free_func: None to use default cleanup
            success = Gst.Meta.register_custom(
                SIMAAI_META_STR,
                "",  # Optional description
                None,  # Default init function
                None   # Default free function
            )

            if not success:
                print(f"ERROR: Failed to register custom metadata: {SIMAAI_META_STR}")
            else:
                print(f"Successfully registered custom metadata: {SIMAAI_META_STR}")
        except Exception as e:
            print(f"Error during metadata registration: {str(e)}")
            traceback.print_exc()

    def request_new_pad(self, templ, direction=None, name=None):
        """
        Handle dynamic pad requests. 
        Pads are created when a new input stream is added dynamically.
        """
        #print("[PYTHON] Requesting new dynamic pad")
        pad_name = f"sink_{len(self.dynamic_pads)}"
        new_pad = Gst.Pad.new_from_template(templ, pad_name)
        self.dynamic_pads.append(new_pad)
        return new_pad
    
    def _get_mpk_json(self):
        """
        - Parses the manifest JSON if present.
        - Finds the plugin that contains the grandparent config (MLA config).
        - Validates if the MPK JSON is present in plugin resources.
        """

        if not self.manifest_json:
            return None

        try:
            plugins = self.manifest_json["applications"][0]["pipelines"][0]["plugins"]

            for plugin in plugins:
                plugin_resources = plugin.get("resources", {})
                plugin_configs = [os.path.basename(x) for x in plugin_resources.get("configs", [])]

                if self.grandparent_config in plugin_configs:
                    # Found grandparent MLA plugin entry in manifest.json
                    # Extract the MPK JSON from `resources -> shared`
                    shared_resources = plugin_resources.get("shared", [])
                    return next((os.path.basename(res) for res in shared_resources if res.endswith(".json")), None)

        except (KeyError, IndexError, TypeError) as e:
            print(f"Error parsing manifest JSON: {e}")

        return None
                    

    
    def _get_plugin_and_config(self, element):
        """
        Get plugin name and config
        """
        if not element:
            return None, None 
        
        plugin_name = element.get_name()
        config_value = None

        if element.list_properties():
            for prop in element.list_properties():
                if prop.name == "config":
                    config_value = element.get_property("config")
                    break
        
        return plugin_name, config_value
    
    def _get_upstream_element(self, pad):
        """
        Get upstream plugin connected to the passed sink pad
        """
        if not pad:
            return None 
        peer_pad = pad.get_peer()

        return peer_pad.get_parent() if peer_pad else None
  
    def _read_json(self, json_path):
        if not os.path.isfile(json_path):
            return None
        
        json_content = None
        with open(json_path, "r") as f:
            json_content = json.load(f)
        
        return json_content


    def do_start(self):
        """
        Handle start even for the aggregator. 
        """
        self.plugin_id = self.get_name()
        for pad in self.iterate_sink_pads():
        #Get the direct upstream plugin (parent)
            parent_element = self._get_upstream_element(pad)
            parent_name, parent_config = self._get_plugin_and_config(parent_element)

            if parent_name:
                print(f"Upstream Plugin Name: {parent_name}")
            if parent_config:
                print(f"Upstream Plugin Config: {parent_config}")

            #If parent config contains "detess_dequant", fetch grandparent info
            if parent_config and "detess_dequant" in parent_config:
                for parent_pad in parent_element.iterate_sink_pads():
                    grandparent_element = self._get_upstream_element(parent_pad)
                    grandparent_name, grandparent_config = self._get_plugin_and_config(grandparent_element)

                    if grandparent_name:
                        print(f"Grandparent Plugin Name: {grandparent_name}")
                    if grandparent_config:
                        self.detess_dequant_pad_name = pad.get_name()
                        self.grandparent_config = os.path.basename(grandparent_config)
                        print(f"Grandparent Plugin Config: {grandparent_config} accessed via sink pad name: {self.detess_dequant_pad_name}")
                        self.mpk_path = os.path.join(PROCESS_MLA_RESOURCES, self._get_mpk_json())
                        print(f"Mpk json path: {self.mpk_path}")

        if self.mpk_path:
            parser = Parser(self.mpk_path)
            self.model_output_shapes, _ = parser.get_tensor_info()
            print(f"Model output shapes: {self.model_output_shapes}")
        stream_start_event = Gst.Event.new_stream_start("aggregator-template-stream")
        self.srcpad.push_event(stream_start_event)
        return True
    
    def finish_buffer(self, buffer):
        """
        Finalizes and pushes the buffer downstream.
        """
        #print("In finish buffer")
        GstBase.Aggregator.finish_buffer(self, buffer)

    def do_set_property(self, property_id, value):

        if property_id == "silent":
            self.silent = value
        elif property_id == "transmit":
            self.transmit = value
        elif property_id == "config":
            self.config = value
        else:
            raise AttributeError(f"Unknown property {property_id}")
    
    def do_get_property(self, property_id):

        if property_id == "silent":
            return self.silent
        elif property_id == "transmit":
            return self.transmit
        elif property_id == "config":
            return self.config
        else:
            raise AttributeError(f"Unknown property {property_id}")

    def extract_metadata(self, buffer: Gst.Buffer) -> None:
        """Input: buffer Gst.Buffer: Input buffer from which metadata will be extracted."""
        meta = buffer.get_custom_meta(SIMAAI_META_STR)
        if not meta or not (s := meta.get_structure()):
            logger.err("No metadata structure found in buffer")
            return None

        try:
            # Extract PCIE info
            ret, pcie_buffer_id = s.get_int64("pcie-buffer-id")
            if ret:
                self.is_pcie = True
                self.in_pcie_buf_id = pcie_buffer_id

            # Extract int64 fields
            for field in ['buffer-id', 'frame-id', 'buffer-offset']:
                ret, value = s.get_int64(field)
                if ret:
                    setattr(self, field.replace('-','_'), value)

            # Extract uint64 fields
            ret, timestamp = s.get_uint64('timestamp')
            if ret:
                self.timestamp = timestamp

            # Extract string fields
            for field in ['stream-id', 'buffer-name']:
                value = s.get_string(field)
                if value:
                    setattr(self, field.replace('-','_'), value)

            metaStruct = MetaStruct(
                frame_id=self.frame_id,
                buffer_name=self.buffer_name,
                timestamp=self.timestamp,
                stream_id=self.stream_id
            )

            # logger.debug(f"Metadata IDs: FrameID:{self.frame_id}, BufferID:{self.buffer_id}, Name:{self.buffer_name}, StreamID:{self.stream_id}")
            # logger.debug(f"Metadata Values: Time:{self.timestamp}, Offset:{self.buffer_offset}, PCIe:{self.is_pcie}, PCIeBufID:{self.in_pcie_buf_id if self.is_pcie else 'N/A'}")
            return metaStruct

        except Exception as e:
            logger.err(f"Failed to extract metadata: {str(e)}")
            return None

    def pack_metadata(self):
        """Pack metadata into bytes"""
        metadata = MetadataStruct()

        # Set fixed fields
        metadata.buffer_id = self.buffer_id
        metadata.frame_id = self.frame_id
        metadata.timestamp = self.timestamp
        metadata.buffer_offset = getattr(self, 'buffer_offset', 0)
        metadata.pcie_buffer_id = getattr(self, 'pcie_buffer_id', 0)

        # Convert strings to bytes and get their lengths
        stream_id_bytes = self.stream_id.encode('utf-8')
        buffer_name_bytes = self.plugin_id.encode('utf-8')

        metadata.stream_id_len = len(stream_id_bytes)
        metadata.buffer_name_len = len(buffer_name_bytes)

        # Pack everything into bytes
        packed_data = bytes(metadata) + stream_id_bytes + buffer_name_bytes

        # logger.debug(f"Packing Base Fields: FrameID:{self.frame_id}, BufferID:{self.buffer_id}, Time:{self.timestamp}, Offset:{metadata.buffer_offset}")
        # logger.debug(f"Packing String Fields: StreamID:{self.stream_id}({metadata.stream_id_len}B), BufferName:{self.plugin_id}({metadata.buffer_name_len}B), PCIeBufID:{metadata.pcie_buffer_id}")
        self.metadata_len = len(packed_data)
        return packed_data

    def insert_metadata_as_header(self, buffer: Gst.Buffer) -> bool:
        """
        Fallback method to insert metadata as buffer header when GstMeta fails.
        Returns: True if successful, False if failed
        """
        try:
            #logger.debug("Falling back to buffer header approach for metadata")
            success, output_map = buffer.map(Gst.MapFlags.WRITE)
            if success:
                try:
                    # Clear output buffer with zeros
                    output_view = memoryview(output_map.data).cast('B')
                    output_view[:] = bytes([0] * len(output_map.data))

                    # Pack metadata
                    metadata_bytes = self.pack_metadata()
                    metadata_size = len(metadata_bytes)

                    # Write metadata size and metadata at the start of buffer
                    struct.pack_into('I', output_map.data, 0, metadata_size)
                    output_view[4:4+metadata_size] = memoryview(metadata_bytes).cast('B')

                    #logger.debug(f"Successfully packed metadata into buffer header. Size: {metadata_size}")
                    return True
                
                except Exception as e:
                    logger.err(f"Error while packing metadata into header: {str(e)}")
                    return False
                finally:
                    buffer.unmap(output_map)
            else:
                logger.err("Failed to map buffer for writing metadata header")
                return False

        except Exception as e:
            logger.err(f"Error in insert_metadata_as_header: {str(e)}")
            traceback.print_exc()
            return False

    def insert_metadata(self, buffer: Gst.Buffer) -> bool:
        """
        Insert metadata into buffer using GstMeta and verify the attachment.
        Returns: True if successful, False if failed
        """
        try:
            # Create new structure
            meta = Gst.Structure.new_empty(SIMAAI_META_STR)
            #logger.debug("Created new metadata structure")

            # Create and set values directly
            values_to_set = {
                'buffer-id': (GObject.TYPE_INT64, self.buffer_id),
                'buffer-name': (GObject.TYPE_STRING, self.plugin_id),
                'buffer-offset': (GObject.TYPE_INT64, 0),
                'frame-id': (GObject.TYPE_INT64, self.frame_id),
                'stream-id': (GObject.TYPE_STRING, self.stream_id),
                'timestamp': (GObject.TYPE_UINT64, self.timestamp)
            }

            if self.is_pcie:
                values_to_set['pcie-buffer-id'] = (GObject.TYPE_INT64, self.pcie_buffer_id)

            # Set all values and verify
            for field_name, (field_type, value) in values_to_set.items():
                meta.set_value(field_name, GObject.Value(field_type, value))
                #logger.debug(f"Set metadata field {field_name} = {value}")

            # Add metadata to buffer
            gst_meta = buffer.add_meta(Gst.Meta.get_info(SIMAAI_META_STR), None)
            if not gst_meta:
                logger.err("Failed to add_meta to buffer")
                return False

            gst_meta.structure = meta
            #logger.debug("Added metadata structure to buffer")

            # Verify metadata attachment
            verified_meta = buffer.get_custom_meta(SIMAAI_META_STR)
            if not verified_meta:
                logger.err("Failed to retrieve metadata after attachment")
                return False

            verified_struct = verified_meta.get_structure()
            if not verified_struct:
                logger.err("Failed to get structure from attached metadata")
                return False

            # Verify each field
            verification_failed = False
            for field_name, (field_type, expected_value) in values_to_set.items():
                #logger.debug(f"Verifying field: {field_name}")

                if not verified_struct.has_field(field_name):
                    #logger.err(f"Field {field_name} is missing from attached metadata")
                    verification_failed = True
                    continue

                actual_type = verified_struct.get_field_type(field_name)
                if actual_type != field_type:
                    #logger.err(f"Field {field_name} type mismatch: expected {field_type}, got {actual_type}")
                    verification_failed = True
                    continue

                if field_type == GObject.TYPE_INT64:
                    success, value = verified_struct.get_int64(field_name)
                elif field_type == GObject.TYPE_UINT64:
                    success, value = verified_struct.get_uint64(field_name)
                elif field_type == GObject.TYPE_STRING:
                    value = verified_struct.get_string(field_name)
                    success = (value is not None)
                else:
                    #logger.err(f"Unsupported field type {field_type} for {field_name}")
                    verification_failed = True
                    continue

                if not success:
                    #logger.err(f"Failed to get value for field {field_name}")
                    verification_failed = True
                    continue

                if value != expected_value:
                    #logger.err(f"Value mismatch for {field_name}: expected {expected_value}, got {value}")
                    verification_failed = True

            if verification_failed:
                logger.debug("Metadata verification failed - values not correctly attached")
                return False

            #logger.debug("Successfully verified all metadata fields")
            return True

        except Exception as e:
            logger.err(f"Error in insert_metadata: {str(e)}")
            traceback.print_exc()
            return False

    def get_gobject_value(self, value, value_type:ValueType) -> GObject.Value:
        
        gvalue = GObject.Value()
        gvalue.init(value_type.value)

        if value_type == ValueType.UINT64:
            gvalue.set_uint64(value)
        elif value_type == ValueType.INT64:
            gvalue.set_int64(value)
        elif value_type == ValueType.STRING:
            gvalue.set_string(value)
        elif value_type == ValueType.DOUBLE:
            gvalue.set_double(value)
        else:
            raise ValueError(f"Unsupported ValueType: {value_type}")

        return gvalue

    def do_aggregate(self, timeout):
        """
        Called when buffers are queued on all sinkpads.
        Calls the run() function defined by the user
        """
        try:
            self.t0 = get_monotonic_timestamp()
            input_pads = []
            #print("In do aggregate")

            # Collect input pads
            pad_iterator = self.iterate_sink_pads()
            while True:
                result, pad = pad_iterator.next()
                if result == Gst.IteratorResult.OK:
                # print(f"Sink Pad: {pad.get_name()}")
                    if pad.get_name().startswith("sink_"):
                        input_pads.append((pad, pad.get_name()))
                elif result == Gst.IteratorResult.DONE:
                    break

            input_buffers = []  #stores list of SimaaiPythonBufer obj
            mapped_buffers = []  # stores mapping of each input buffer to it's map
            logger.info(f"Number of pads: {len(input_pads)}")

            # Process input buffers
            for pad, pad_name in input_pads:
                if pad.is_linked():
                    buffer = pad.pop_buffer()
                    if buffer:
                        metaObj = self.extract_metadata(buffer)
                        success, buffer_map = buffer.map(Gst.MapFlags.READ)
                        if pad_name == self.detess_dequant_pad_name:
                            reshape_tensors = self.model_output_shapes
                        else:
                            reshape_tensors = None    
                        input_buffers.append(SimaaiPythonBuffer(metadata=metaObj, map=buffer_map, reshape_tensors = reshape_tensors))
                        mapped_buffers.append((buffer, buffer_map))

            # Create output buffer
            output_buffer = Gst.Buffer.new_allocate(None, self.out_size, None)
            if not output_buffer:
                print("Failed to allocate output buffer")
                return Gst.FlowReturn.ERROR

            # Map output buffer after metadata is set
            success, output_map = output_buffer.map(Gst.MapFlags.WRITE | Gst.MapFlags.READ)
            if not success:
                print("Failed to map output buffer")
                return Gst.FlowReturn.ERROR

            # Try to insert metadata using GstMeta first
            if not self.insert_metadata(output_buffer):
                logger.debug("GstMeta insertion failed, attempting buffer header approach")
                self.metadata_add_failed = True


            if self.metadata_add_failed and self.next_plugin_is_metaparser:
                if not self.insert_metadata_as_header(output_buffer):
                    logger.error("insert metadata as a header failed")
                    return Gst.FlowReturn.ERROR
                logger.warning("Successfully inserted metadata using header approach, Meta Parser should be used as a next plugin")
            else:
                logger.warning("Pygast Plugin is handling the overlay by default (MetaParser plugin need  not be used as next)")

            # call to the run function - main logic
            if input_buffers:
                if self.metadata_add_failed and self.next_plugin_is_metaparser:
                    # move the buffer pointer after header(4 bytes to hold metadata size) + packed metadata bytes length
                    data_start = 4 + self.metadata_len
                    output_data = memoryview(output_map.data)[data_start:]
                else:
                    output_data = output_map.data

                self.run(input_buffers, output_data)

            # unmap all input maps
            for buffer, buffer_map in mapped_buffers:
                buffer.unmap(buffer_map)

            #unmap output buffer
            output_buffer.unmap(output_map)

            # Finish buffer
            self.finish_buffer(output_buffer)

            # Handle KPI reporting
            self.t1 = get_monotonic_timestamp()
            elapsed_time = (self.t1 - self.t0) / 1000.  #convert to millisecond

            if self.transmit:
                kernel_start, kernel_end = 0,0
                kpi_struct = Gst.Structure.new_empty("kpi")
                kpi_struct.set_value("plugin_start", self.get_gobject_value(self.t0, ValueType.UINT64))
                kpi_struct.set_value("plugin_end", self.get_gobject_value(self.t1, ValueType.UINT64))
                kpi_struct.set_value("duration", self.get_gobject_value(elapsed_time, ValueType.DOUBLE))
                kpi_struct.set_value("kernel_start", self.get_gobject_value(kernel_start, ValueType.UINT64))
                kpi_struct.set_value("kernel_end", self.get_gobject_value(kernel_end, ValueType.UINT64))
                kpi_struct.set_value("frame_id", self.get_gobject_value(self.frame_id, ValueType.INT64))
                kpi_struct.set_value("plugin_id", self.get_gobject_value(self.plugin_id, ValueType.STRING))
                kpi_struct.set_value("plugin_type", self.get_gobject_value(PLUGIN_CPU_TYPE, ValueType.STRING))
                kpi_struct.set_value("stream_id", self.get_gobject_value(self.stream_id, ValueType.STRING))

                message = Gst.Message.new_application(self, kpi_struct)
            # Post the message to the bus
                bus = self.get_bus()
                if bus:
                    bus.post(message)
                else:
                    Gst.error("Could find a bus to send")

            return Gst.FlowReturn.OK

        except Exception as e:
            print(f"Error in do_aggregate: {str(e)}")
            return Gst.FlowReturn.ERROR

    def run(self, input_buffers: List[Gst.Buffer], output_buffer: Gst.Buffer) -> None:
        """
        Input:  
        input_buffers: List[Gst.Buffer] List of input buffers, source from each pad.  
        output_buffer: Gst.Buffer Output buffer that needs to be overwritten.  

        Implement your logic within this function. Process the input buffers, and modify the output  
        buffer.
        """

        raise NotImplementedError("run def not implemented")