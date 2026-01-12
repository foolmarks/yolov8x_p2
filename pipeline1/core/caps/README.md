# Description
Caps library is a generic library for handling caps negotiation in gstreamer plugins. This library allows to create pads with specific capabilities and update fields in config files during caps negotiation.

# Caps config description
For library to work config file has to have `caps` section in config file. This block contains 2 arrays, that describe caps for pads: `sink_pads` or `src_pads`.

Each object for `sink_pads` or `src_pads` arrays have same structure that contains:

- `media_type` – media type of caps. Please refere to GStreamer documentation to understand media types
- `params` – array of objects that describes parameters in this media type. 
  Each object consist of:
	- `name` – name of parameter
	- `type` – type of parameter. 
	Supported types: `int`, `float`, `string`
	- `json_field` – name of `JSON` field to update.
	Supported types: parameter name, or `null` if no saving of value is needed
	If param is part of `sink_pads`, than this field will be updated with value provided by upstream plugin in fixated caps
	If param is part of `src_pads`, than value from this field will be used to fixate src caps
	- `values` – valid values for parameter. Can be in form of range if 2 values are separated as `-`, or in form of set of valid values, if they are separated by `,`. If 1 value provided - it will act as fixed caps.

Example of pad object:
```JSON
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
            "type": "float",
            "name": "height",
            "values": "10.80",
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
```
In this case `width` is integer range from `4` to `4096`, `height` is fixed float value `10.80`, and format is range of string values.

Example above will be converted in pad template with caps:
`video/x-raw, width=(int)[4, 4096], height=(float)10.80, format=(string){I420, NV12, RGB, BGR, Grayscale};`

# Example of full caps block
```JSON
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
```
