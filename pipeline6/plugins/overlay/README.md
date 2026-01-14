# Overlay
A GStreamer plugin that draws bounding boxes/poses/points/etc on the original image. In design documentation it also may be called `Annotator`.

## Table of Contents

- [Overlay](#overlay)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Rendering engine](#rendering-engine)
      - [Setting up the rendering rules](#setting-up-the-rendering-rules)
      - [Labels file](#labels-file)
  - [Configuration](#configuration)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Introduction

This plugin is usually used draw bounding boxes, poses or other simple objects on the original image (frame). This plugin is nested from AGGREGATOR type.

## Features

- YUV420 format supporting only (for input and output types)
- Includes header only lightweight library for object rendering.
- Easy to extend

## Getting Started

No specific steps needed. As an any other plugin, this one can be built via Yocto SDK:
```
source /opt/poky/4.0.10/environment-setup-cortexa65-poky-linux
mkdir build && cd build && cmake .. && make
```

### Prerequisites
- Yocto SDK installed

### Installation

For manual installation:
```
scp libgstsimaaioverlay.so sima@<IP address of DaVinci board>:/data/your_pipeline/libs
```

## Usage

### Properties
Unlike the other plugins, `overlay` does not require `config.json` file. But it requries a few command-line parameters provided. See the example below:
```
... \
! simaai-overlay2 render-info="input::allegrodec,bbox::nms" labels-file="/data/coco.names" silent=true dump-data=false name=overlay \
! ...
```
- `render-info` - contains an information about input buffers names and type of input data. See the section below with detailed explanation.
- `labels-file` - path to file with COCO dataset class names.
- `dump-data` - dump render data to log files under the /tmp directory

### Caps

Plugin supports 2 sink and 1 src pad template:

 - `sink_application_data` - data pad. Source of data, that should be overlayed on top of original image
    - provides render rule, that need to be used for overlaying
 - `sink_in_img_src` - image pad. Source of image, where data from previous pad should be overlayed
    - provides image resolution and format
 - `src` - output pad. Overlayed images pushed here
    - provides to next plugin format and resolution of output image


### Rendering engine
Internally, `overlay` plugin uses a lightweight library for rendering graphical primitives such as points, lines, etc. Under the hood this library utilizes `opencv` library which is already included into Yocto SDK. `ext` folder contains the source code of this engine.

#### Setting up the rendering rules
As mentioned above, `overlay` plugin is an AGGREGATOR, which means it may receive more than one input buffers at the time. One of these buffers **must be** an input image (frame). The command line option called `render-info` is used for that. It shall contain a comma-separated pairs, where key - is the name of callback function defined [here](https://bitbucket.org/sima-ai/gst-simaai-plugins-base/src/develop/gst/overlay/ext/yuv420.cc) and value is the input buffer name.
> THERE IS A CONSTRAINT HERE. `render-info` parameter **must have** `input::<buffername>` pair. In other words, `input` key is reserved for the buffer that contains an original image.

For example: `render-info="input::allegrodec,bbox::nms"` line says to `overlay` plugin to use buffer with name `allegrodec` as a source of original frame, and buffer with name `nms` as source of bounding boxes.
> USER CAN DEFINE MORE THAN ONE SOURCE IN CASE OF COMPLEX PIPELINE. BUT THE CONSTRAINT HERE IS TO HAVE input::buffername PAIR

Parameter `render-info` itself does not force `overlay` to start drawing objects on top of the image.
Another parameter called `render-rules` says to overlay what exactly has to be rendered on top of original image. It contains `|` separated names of callbacks defined at https://bitbucket.org/sima-ai/gst-simaai-plugins-base/src/develop/gst/overlay/ext/yuv420.cc which has to be applied to associate input buffers.
For example the following string:
```
... ! simaai-overlay2 name=overlay width=1280 height=720 format=2 render-info="input::allegrodec,bbox::a65-topk,text::a65-transforms" render-rules="bbox|text" ! allegroenc2 config="something.json"
```
says to `overlay`:
- take the original image (frame) from the buffer with name `allegrodec`
- take the bounding box data from the buffer with name `a65-topk`
- take the image classification text from the buffer with name `a65-transforms`
- Apply callback function with name `"bbox"` (which draws Bounding Boxes) with the information taken from `a65-topk` buffer
- Apply callback function with name `"text"` (which draws Bounding Boxes) with the information taken from `a65-transforms` buffer

All available callbacks are listed in `yuv420.cc` file. Currently plugin supports:
  - "bbox" - just bounding boxes
  - "bboxt" - bounding boxes + tracker specifc data
  - "bboxd" - bounding boxes + distance calculation
  - "text" - text drawing
  - "pose" - pose rendering (OpenPose specific)
  - "bboxs" - bounding boxes + score data

> USER CAN EASILY ADD ITS OWN CALLBACKS TO DRAW PIPELINE SPECIFIC DATA (SEE SECTION Contribution for details)

#### Labels file
If your pipeline does some classification, most likely you want to display the class name on top of bounding box. To do that, you need to provide a path to a file with class names (usually it contains a classes from COCO dataset). The example of such file could be found [here](https://bitbucket.org/sima-ai/vdp-app-config-default/src/develop/applications/Ebike/resources/coco.names). In short there should be the file with class names, where each class name is placed on the new line of file. Usually such label file has to be provided along with model to proper match the class_id in output tensor with the class name in file.
Please refer to [Setting up the rendering rules](####Setting-up-the-rendering-rules) page to get the list of available callbacks and check their source code to know which ones support labeling.


## Configuration

No configuration files needed for this plugin.

## Contributing

User can add it's own callbacks in order to support new pipeline or new types of data. To do that:
- define new callback function for `simaai::overlay::Yuv420` class
- register it in `simaai::overlay::Yuv420` class constructor
- re-compile plugin and run it!
- Submit your PR for `https://bitbucket.org/sima-ai/gst-simaai-plugins-base/src/develop/gst/overlay/ext/` so anyone else can start using your callback.

Also feel free to update the rest of plugin code to start supporting more image types and more:
https://bitbucket.org/sima-ai/gst-simaai-plugins-base/src/develop/gst/overlay/

If you need to update OpenCV based functions, used by `ext` library, submit the PR here:
- https://bitbucket.org/sima-ai/a65-apps/src/develop/cv_helpers/
- https://bitbucket.org/sima-ai/a65-apps/src/develop/cv_operators/

## License


## Acknowledgments

