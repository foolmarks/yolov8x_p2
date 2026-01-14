# simaaiprocessmla

Plugin is used to run data thought model on MLA cores. This plugin has to be used for all MLA operations in the pipeline.
User must not change plugin source code in any way. Only configuration changes are needed to start using this plugin.

Under the hood, this plugin uploads `model.elf` provided by user and sends a request of processing input data to the MLA processor using uploded model.

## Table of Contents

- [simaaiprocessmla](#simaaiprocessmla)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Plugin properties](#plugin-properties)
  - [Configuration](#configuration)
  - [Caps](#caps)
  - [Usage](#usage)
  - [Config File Example](#config-file-example)

## Requirements

1. Upstream plugins, that are linked to `simaaiprocessmla` sink pads, should **produce buffers with segmented memory allocator**. When using upstream sima plugins, please use `gst-inspect-1.0` to review the use of `sima-allocator-type` parameter option to enable segment allocator;
2. For proper work of `simaaiprocessmla` upstream plugin should work on `src pad` with caps, that are described in `sink pads` block in plugin config. If `CAPS_ANY` used in plugin – capsfilter is needed between `simaaiprocessmla` and upstream plugin.

## Plugin properties

For up to date properties list and description, please, refer to `gst-inspect-1.0` output

List of properties:

- `dump-data` – Flag to save output buffers in binary dumps in /tmp/{node-name}-{frame_id}.out
Valid range: `false`, `true`
Default: `false`
- `name` – The name of the object. Also used as name of output buffer (`buffer-name` metadata field)
Default: `simaaiprocesscvu%d` where `%d` is instance number
- `num-buffers` – Number of buffers to be allocated in GstBufferPool
Valid range: `1 - 4294967295`
Default: `2` 
- `silent` – Flag to produce verbose output (silent=false – produce output)
Valid range: `false`, `true`
Default: `true`
- `transmit` – Flag to transmit KPI Messages
Valid range: `false`, `true`
Default: `false`
- `config` – Path to the JSON config file with instance configurations
Default: (empty)
- `multi-pipeline` – Flag to turn on/off support of multiple separate pipelines launched at the same time
Valid range: `false`, `true`
Default: `false`


## Configuration

Each instanse has to have it's own [`config.json`](#config-file-example) which consist of such important fields:

- `caps` – block, that describes input and output capabilities for specific model. Refer to [Caps](#caps) section for detailed description of this block;
- `model_path` – path to the model used in this plugin instance;
- `outputs` – array of segments of model output.

Each object in `outputs` array will consist of:

- `name` – name of segment;
- `size` – size of segment.

`simaaiprocessmla` will allocate 1 contiguous output, that is logically devided to segments described in `outputs` array. This buffer will have size of all segments combined **without any padding**, so user should make sure, that both **segments sizes and total size will be aligned to 16 bytes** so that all processors could properly work with this buffer

For example of config file, please, go to [Config File Example](#config-file-example) section.

## Caps

Caps of each instance are described in `caps` block in [`config.json`](#config-file-example)
This block is generic accros multiple plugins and consists of 2 arrays:

- `sink_pads` - describes input capabilities of model
- `src_pads` - describes output capabilities of model

**NOTE:** Both arrays, in case of `simaaiprocessmla`, will work only with 1 src and sink pad (because of the design of plugin, that can be connected only to 1 upstream and to 1 downstream plugin)

For more detailed description of caps block, please, refer to Caps Negotiation Library README.md. Caps sections from example [`config.json`](#config-file-example) can be converted to caps:

- sink caps – `video/x-raw, width=(int)[1, 4096], height=(int)[1, 4096], format=(string){RGB, BGR}`
- src caps – `application/vnd.simaai.tensor, format=(string)MLA`

## Usage

Example `gst-string` with `simaaisrc`
(works on `src pad` with `CAPS ANY`):
```BASH
simaaisrc location=simaai_preproc-001.out node-name="simaai_preproc" mem-target=1 \
! 'video/x-raw, width=(int)480, height=(int)480, format=(string)RGB' \
! simaaiprocessmla name='simaai_mla_centernet' config=/data/mla.json num-buffers=5 \
! fakesink
```

P.S. Caps filter between `simaaisrc` and `simaaiprocessmla` is needed because first one works with `CAPS_ANY`. In other cases (if upstream plugin provides right caps) - caps filter is not needed

## Config File Example
```JSON
{
  "version" : 0.1,
  "simaai__params" : {
    "next_cpu" : 1,
    "outputs" : [
        {
          "name": "hm_tensor",
          "size": 115200
        },
        {
          "name": "paf_tensor",
          "size": 172800
        }
    ],
    "batch_size" : 1,
    "batch_sz_model": 1,
    "model_path" : "/data/openpose.elf"
  },
  "caps": {
    "sink_pads": [
      {
        "media_type": "video/x-raw",
        "params": [
          {
            "name": "format",
            "type": "string",
            "values": "RGB, BGR",
            "json_field": null
          },
          {
            "name": "width",
            "type": "int",
            "values": "1 - 4096",
            "json_field": null
          },
          {
            "name": "height",
            "type": "int",
            "values": "1 - 4096",
            "json_field": null
          }
        ]
      }
    ],
    "src_pads": [
      {
        "media_type": "application/vnd.simaai.tensor",
        "params": [
          {
            "name": "format",
            "type": "string",
            "values": "MLA",
            "json_field": null
          }
        ]
      }
    ]
  }
}
```
