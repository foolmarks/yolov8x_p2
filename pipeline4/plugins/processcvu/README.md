# simaaiprocesscvu

Plugin is used to run a graph on the CVU cores. This plugin has to be used for all EVXX operations in the pipeline.
User must not change plugin source code in any way. Only configuration changes are needed to start using this plugin.

Under the hood, this plugin sends a request of processing input data to the specific EVXX graph. Refer [SiMa.ai SoC Software Graph Documentation](https://sw-web.eng.sima.ai/graphs/latest/) to see the list of available graphs that can be used by the pipeline.

## Table of Contents

- [simaaiprocesscvu](#simaaiprocesscvu)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Plugin properties](#plugin-properties)
  - [Configuration](#configuration)
	- [Graph parameters](#graph-parameters)
	- [Segment to buffer mapping blocks](#segment-to-buffer-mapping-blocks)
	- [Caps block](#caps-block)
  - [Usage](#usage)
  - [Config file example](config-file-example)

## Requirements

1. Upstream plugins, that are linked to `simaaiprocesscvu` sink pads, should **produce buffers with segmented memory allocator**. When using upstream sima plugins, please use `gst-inspect-1.0` to review the use of `sima-allocator-type` parameter option to enable segment allocator;
2. For proper work of `simaaiprocesscvu` upstream plugin should work on `src pad` with one of caps, that are described for `sink pads` in plugin config. If `CAPS_ANY` used in plugin – you should use capsfilter between `simaaiprocesscvu` and upstream plugin

## Plugin properties

For up to date properties list and description, please, refer to `gst-inspect-1.0` output

List of properties:

- `name` – The name of the object. Also used as name of output buffer (`buffer-name` metadata field).
Default: `simaaiprocesscvu%d`, where `%d` is instance number in pipeline;
- `dump-data` – Flag to save output buffers in binary dumps in `/tmp/{name-of-the-object}-:03{frame_id}.out`.
Valid range: `false`, `true`.
Default: `false`;
- `num-buffers` – Number of buffers to be allocated in GstBufferPool.
Valid range: `1 - 4294967295`.
Default: `5`;
- `silent` – Flag to produce verbose output (silent=false – produce output).
Valid range: `false`, `true`.
Default: `true`;
- `transmit` – Flag to transmit KPI Messages.
Valid range: `false`, `true`.
Default: `false`;
- `config` – Path to the JSON config file with instance configurations.
Default: `/mnt/host/evxx_pre_proc.json`.


## Configuration

Each graph has it’s specific configuration parameters, that has to be reflected in `config.json`. Configurations consist of three major parts:

- [Graph parameters](#graph-parameters)
– EVXX graph specific parameters;
- [Segment to buffer mapping blocks](#segment-to-buffer-mapping-blocks) (`input_buffers`&`output_memory_order`)
– describe mapping of input segments of buffers to graph inputs and order of segments in output buffer;
- [Caps block](#caps-block)
– describe caps for sink/src pads and how they should be parsed.

### Graph parameters

All graph parameters are described for each graph in [SiMa.ai SoC Software Graph Documentation](https://sw-web.eng.sima.ai/graphs/latest/). But some parameters are common for all graphs.

Here is the list of generic EV parameters:

- `graph_name` – Name of EVXX graph.
Should be the same as desired graph on [Graph Documentation](https://sw-web.eng.sima.ai/graphs/latest/) page;
- `next_cpu` – CPU of next plugin in pipeline.
Supported values: `MLA`, `CVU`, `A65`;
- `cpu` – CPU of this plugin.
Should always be `CVU`;
- `debug` – debug levels for EVXX graph.
Supported values – need double check with specific graph documentation, but generic values are: `EVXX_DBG_DISABLED`, `EVXX_DBG_LEVEL_1`, `EVXX_DBG_LEVEL_2`, `EVXX_DBG_LEVEL_3`.

Graph parameters example can be found in [example config.json](config-file-example)

### Segment to buffer mapping blocks

In `config.json` we have 2 blocks, that describe input and output mapping: `input_buffers` and `output_memory_order`.

`input_buffers` – describes input buffers and mapping of their segments to dispatcher. Block is array of objects. Each object describe input buffer from one of sink pad. Each buffer has 2 properies:

- `name` – a set of valid buffer names separated by comma. `buffer-name` produced by upstream plugin should be the same as one of names described in this field 
- `memories` – array of objects, that describe mapping of segments of this buffer to graph inputs. Objects in this block consists of:
  - `segment_name` – name of segment in input buffer
  - `graph_input_name` – name of mapped graph input (Reffere to [Input Buffer details section of used Graph](https://sw-web.eng.sima.ai/graphs/latest/))

`output_memory_order` – order of graph output buffers. All graph outputs and intermediate buffers should be described here in desired order. ConfigManager will internally provide size for each graph buffer. Size of plugin output buffer will be calculated as total of all output & intermediate buffers. Plugin output buffer will be logically divided into segments with sizes of each graph output and intermediate buffer in order specified by user

Actual segment to buffer mapping blocks example can be found in [example config.json](config-file-example).
Generic segment to buffer mapping blocks example:
```JSON
"input_buffers": [
    {
        "memories": [
            {
                "segment_name" : "tensor_one",
                "graph_input_name" : "input_one",
            {
            }
                "segment_name" : "tensor_two",
                "graph_input_name" : "input_two",
            }
        ]
        "name": "buffer_one"
    },
	{
        "memories": {
            "tensor": "input_three"
        },
        "name": "buffer_two, buffer_three, buffer_four"
    }
],
"output_memory_order" : [
	"output_tensor_3",
	"output_tensor_1",
	"output_tensor_2"
]
```

### Caps block

In `config.json` we have object called `caps`. This object consist of 2 arrays:

- `sink_pads` – array of objects, where each object describe **caps for separate sink pad** 
- `src_pads` – array of objects, where each object describe **different valid caps for single src pad**

For more detailed description of caps block, please, refer to Caps Negotiation Library README.md. 

Caps block from [example config.json](config-file-example) can be converted to caps:

- sink caps – `video/x-raw, width=(int)[1, 4096], height=(int)[1, 4096], format=(string){I420, NV12, RGB, BGR, Grayscale}`
- src caps – `video/x-raw, width=(int)[1, 4096], height=(int)[1, 4096], format=(string){RGB, BGR}`

## Usage

Example `gst-string` with `simaaidecoder` 
(works on `src pad` with same caps, described in [example config.json](config-file-example)):
```BASH
rtspsrc location=<RTSP_SRC> ! rtph264depay wait-for-keyframe=true ! h264parse \
! 'video/x-h264,parsed=true,stream-format=(string)byte-stream,alignment=(string)au,width=[4,4096],height=[4,4096]' \
! simaaidecoder name='decoder' dec-fmt='YUV420P' next-element='CVU' sima-allocator-type=2 \
! simaaiprocesscvu_new name='simaai_preproc' config='/data/preproc.json' num-buffers=5 \
! fakesink
```

Example `gst-string` with `simaaisrc` 
(works on `src pad` with `CAPS ANY`):
```BASH
simaaisrc location='input_image.out' node-name="decoder" mem-target=1 \
! 'video/x-raw, width=(int)1280, height=(int)720, format=(string)I420' \
! simaaiprocesscvu_new name='simaai_preproc' config='/data/preproc.json' num-buffers=5 \
! fakesink
```

### Config File Example

```JSON
{
    "graph_name": "gen_preproc",
    "next_cpu": "MLA",
    "cpu": "CVU",
    "debug": "EVXX_DBG_DISABLED",
    "aspect_ratio": false,
    "batch_size": 1,
    "channel_mean": [
        0.5,
        0.5,
        0.5
    ],
    "channel_stddev": [
        1.0,
        1.0,
        1.0
    ],
    "input_depth": 3,
    "input_height": 720,
    "input_img_type": "IYUV",
    "input_stride": 1,
    "input_width": 1280,
    "normalize": true,
    "output_depth": 3,
    "output_dtype": "EVXX_INT8",
    "output_height": 480,
    "output_img_type": "RGB",
    "output_stride": 1,
    "output_width": 480,
    "padding_type": "CENTER",
    "q_scale": 256.2445112693853,
    "q_zp": 0,
    "scaled_height": 480,
    "scaled_width": 480,
    "scaling_type": "BILINEAR",
    "tessellate": true,
    "tile_depth": 3,
    "tile_height": 24,
    "tile_width": 96,
    "input_buffers": [
        {
            "memories": [
                {
                    "segment_name" : "parent",
                    "graph_input_name" : "input_image"
                }
            ],
            "name": "decoder"
        }
    ],
    "output_memory_order": [
        "output_tessellated_image",
        "output_rgb_image"
    ],
    "caps": {
        "sink_pads": [
            {
                "media_type": "video/x-raw",
                "params": [
                    {
                        "type": "int",
                        "name": "width",
                        "values": "4 - 4096",
                        "json_field": "input_width"
                    },
                    {
                        "type": "int",
                        "name": "height",
                        "values": "4 - 4096",
                        "json_field": "input_height"
                    },
                    {
                        "type": "string",
                        "name": "format",
                        "values": "I420, NV12, RGB, BGR, Grayscale",
                        "json_field": "input_img_type"
                    }
                ]
            }
        ],
        "src_pads": [
            {
                "media_type": "video/x-raw",
                "params": [
                    {
                        "name": "width",
                        "type": "int",
                        "values": "1 - 4096",
                        "json_field": "output_width"
                    },
                    {
                        "name": "height",
                        "type": "int",
                        "values": "1 - 4096",
                        "json_field": "output_height"
                    },
                    {
                        "name": "format",
                        "type": "string",
                        "values": "RGB, BGR",
                        "json_field": "output_img_type"
                    }
                ]
            }
        ]
    }
}
```
