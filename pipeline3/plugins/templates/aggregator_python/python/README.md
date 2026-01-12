# PyGast Plugin Template ( Python Plugin Template)

## Table of Contents
- [PyGast Plugin Template](#pygast-plugin-template)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
    - [Metadata Handling](#metadata-handling)
    - [Buffer Management](#buffer-management)
    - [Error Handling](#error-handling)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Plugin Architecture](#plugin-architecture)
    - [Metadata Structure](#metadata-structure)
    - [Buffer Handling](#buffer-handling)
    - [Pipeline Examples](#pipeline-examples)
  - [Custom Plugin Development](#custom-plugin-development)
    - [Plugin Template Structure](#plugin-template-structure)
    - [Implementing Custom Logic](#implementing-custom-logic)
    - [Example Implementation](#example-implementation)
  - [Memory Management](#memory-management)
  - [Logging](#logging)
  - [Error Handling](#error-handling-1)
  - [Contributing](#contributing)

## Introduction
The PyGast Plugin Template(python_plugin_template.py) provides a framework for creating GStreamer plugins in Python (Custom PyGast Plugin). It handles dynamic pad creation, metadata management, and buffer processing while allowing developers to focus on implementing their specific plugin logic.

## Features

### Metadata Handling
- Dual-mode metadata attachment:
  * Primary: GstCustomMeta attachment using `Gst.Meta.register_custom`
  * Fallback: Header-based metadata insertion in the output buffer itself
- Automatic metadata registration and verification
- Support for various metadata fields including:
  * Buffer ID, Frame ID, Timestamp
  * Stream ID and Buffer Name
  * PCIe buffer information

### Buffer Management
- Dynamic input pad creation
- Automated buffer mapping and unmapping
- Support for multiple input streams
- Custom output buffer size handling

### Error Handling
- Comprehensive error tracking and reporting
- Graceful fallback mechanisms
- Detailed logging and debugging capabilities


## Usage

### Plugin Architecture
The template provides a base class `AggregatorTemplate` that handles:
- Dynamic pad creation
- Buffer aggregation
- Metadata management
- Memory mapping

### Metadata Structure
```python
class MetadataStruct(ctypes.Structure):
    _fields_ = [
        ("buffer_id", ctypes.c_int64),
        ("frame_id", ctypes.c_int64),
        ("timestamp", ctypes.c_uint64),
        ("buffer_offset", ctypes.c_int64),
        ("pcie_buffer_id", ctypes.c_int64),
        ("stream_id_len", ctypes.c_uint32),
        ("buffer_name_len", ctypes.c_uint32),
    ]
```

### Buffer Handling
Output buffer structure:
```
[Metadata Size (4 bytes)][Metadata][Actual Data]
```

### Pipeline Examples

> **IMPORTANT**: Custom PyGast Plugins must always be paired with the simaaimetaparser plugin in the pipeline. The simaaimetaparser plugin is responsible for proper metadata handling and passing input buffers from Pygast plugins to the downstream.

```bash
# Basic usage
gst-launch-1.0 videotestsrc ! custom-pygast-plugin ! simaaimetaparser ! downstream-plugin

# With metadata debugging
GST_DEBUG=4 gst-launch-1.0 videotestsrc ! custom-pygast-plugin ! \
    simaaimetaparser silent=false dump-data=true ! downstream-plugin

# Multiple input streams with performance monitoring
gst-launch-1.0 videotestsrc ! custom-pygast-plugin name=myplugin \
    videotestsrc ! myplugin. \
    myplugin. ! simaaimetaparser transmit=true ! downstream-plugin

# Pipeline with all debug options
GST_DEBUG="custom-pygast:4,simaaimetaparser:4" gst-launch-1.0 \
    videotestsrc ! custom-pygast-plugin ! \
    simaaimetaparser silent=false dump-data=true transmit=true ! \
    downstream-plugin
```

Note: The simaaimetaparser plugin provides several useful options:
- `silent=false`: Enables detailed debug output
- `dump-data=true`: Dumps buffer data for debugging
- `transmit=true`: Enables performance monitoring and KPI transmission

## Custom Plugin Development

### Plugin Template Structure
```python
class MyPlugin(AggregatorTemplate):
    def __init__(self):
        super(MyPlugin, self).__init__(plugin_name="my_plugin", out_size=output_size)
        # Initialize your plugin-specific variables

    def run(self, input_buffers, output_buffer):
        # Implement your plugin logic here
        pass
```

### Implementing Custom Logic
1. Subclass `AggregatorTemplate`
2. Define your plugin's output buffer size
3. Override the `run()` method with your processing logic
4. Handle input buffers and write to output buffer

### Implementation steps
## Process input buffers
## processing logic
## Write to output buffer
## Outpur Buffer size
- Consider metadata size in output buffer calculations:
  ```python
  total_size = metadata_size + max_string_size (to store stream_id & buffer_name) + output_size
  ```

## Logging
The template provides a comprehensive logging system:
```python
logger.set_level(LogLevel.ERR)  # Set logging level
logger.info("Informational message")
logger.err("Error message")
logger.debug("Debug message")
```

Log Levels:
- EMERG (0)
- ALERT (1)
- CRIT (2)
- ERR (3)
- WARNING (4)
- NOTICE (5)
- INFO (6)
- DEBUG (7)

## Error Handling
The template includes robust error handling:
- Exception catching and logging
- Metadata verification
- Buffer mapping verification
- Graceful fallback mechanisms


## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request to https://bitbucket.org/sima-ai/gst-simaai-plugins-base/src/Edgematic-release-1.4.0/gst/templates/aggregator_python/python/
  or https://bitbucket.org/sima-ai/gst-simaai-plugins-base/src/develop/gst/templates/aggregator_python/python/ depends on your requirement

## License


## Acknowledgments
- SiMa.ai team for GStreamer infrastructure
- PyGast plugin developers