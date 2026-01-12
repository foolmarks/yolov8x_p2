# Sima Custom Source

This plugin is similar to `filesrc` or `multifilesrc` plugins but supports SiMa memory allocation. So it can be used to debug/develop SiMa pipelines.

## Table of Contents

- [Sima Custom Source](#sima-custom-source)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Parameters explanation](#parameters-explanation)
  - [Configuration](#configuration)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Introduction

This plugin reads a given file from the file system and puts it into SiMa memory, so that another SiMa plugins (like `process2` or `process-cvu`) can read it from SiMa memory.

## Features

- Works as `multifilesrc` but does SiMa memory allocation
- Supports A65, EV74, MLA memory
- Internally uses [SiMa memory library](https://bitbucket.org/sima-ai/simaai-memory-lib/src/develop/)

## Getting Started

### Prerequisites

- Yocto SDK installed

### Installation

For custom installation:
```
scp libgstsimacustomsrc.so sima@<IP address of DaVinci board>:/data/your_pipeline/libs
```

## Usage

This plugin is extremely useful if you're debugging or developing new pipeline. `simaaisrc` plugin allows you to mock the input for your plugin and therefore test the another plugin in isolation. See examples below to get an idea how it works:

Assuming you have a pipeline that looks like:
```
input_image -> decoder -> pre-process[EV74] -> process[MLA] -> post-process[EV74] -> nms[A65] -> output
```

And there is a need to debug NMS plugin because for some reason it does not provide to you valid output.

For debugging sake it is importan to preserve deterministic input, so developer can check NMS plugin work step by step using GDB or any other tool. 

To do that, we don't really need the rest of plugins in the pipeline (like decoder, process-cvu, process2, etc). Instead, we need **to mock** the input by using `simaaisrc` plugin and simplify pipeline to:
```
simaaisrc -> nms -> fakesink
```

Let's see how it will look like in real world:
- First of all we need to get the output for the previous plugin (in our case it could be a `process-cvu` plugin that runs post processing EV graph). To dump the output of it you need to set `dump: 1` in `post-process.json` config file.
- Once you did that - run your pipeline as usual, without any other modifications for few seconds. After that you should be able to see `ev-post-process-XXX.out` files at `/tmp` folder on the board.
- Take one file (for example `ev-post-process-001.out`) and copy to somewhere to `/data`. This step is needed, because `/tmp` is not a persistent storage and after board reboot all data will disappear and you'll need to get the dumps again.
- Now you're ready to start using simaaisrc plugin:
  ```
  gst-launch-1.0 gst-plugin-path=/path/plugins/folder simaaisrc location=/data/ev-post-process-001.out node-name="ev-cnet-postproc" blocksize=786432 delay=1000 mem-target=0 ! fakesink
  ```
### Parameters explanation

- `location` - an absolute path to the file which will be used as an input
- `node-name` - a name of the plugin we're mocking up (can be taken from it's `config.json`).
- `blocksize` - size of input file in bytes
- `delay` - delay before buffer is pushed to next plugin in milliseconds
- `loop` [default false]- loop property controls whether the plugin should repeatedly read the input files in a loop or stop after processing them once.
- `mem-target` - type of SiMa memory that will be used for allocation. Can be `0, 1, 2`
  - `0` - SIMAAI_MEM_TARGET_GENERIC (for A65)
  - `1` - SIMAAI_MEM_TARGET_EV74 (for EV74)
  For more information about memory targets and memory allocations please refer to:
  - https://bitbucket.org/sima-ai/simaai-memory-lib/src/develop/
  - https://bitbucket.org/sima-ai/simaai-soc-pipeline/src/develop/core/allocator/
- `segments` - forces an element to allocate a memory using a segment memory API. See the section below for detailed explanation

#### Using segment memory API
Segment Memory API provides a mechanism to allocate a logical parts of contigous buffer with their own unique names instead of allocating entire buffer. This mechanism is extremely useful when User wants to debug/troubleshoot the `processcvu_new` element. This element uses a new Dispatcher API, that requires memory segments as an input. In this scenario, User can use a separate files that corresponds to each segments. Lets take an example:

OpenPose post-process graph `pose_postproc` requires two tensors that represents HM (Heat Map) and PAF (Partial affinity fields). To do that User can run the following GST string:
```
gst-launch-1.0 simaaisrc segments='hm_tensor=/data/hm_tensor_in,paf_tensor=/data/paf_tensor_in' node-name=simaaiprocessmla0 mem-target=1 delay=30 ! 'application/vnd.output-mla' ! simaaiprocesscvu_new name='simaai_postproc' num-buffers=5 config='/data/mem_segments/config_postproc.json' dump_data=true ! 'application/vnd.openpose-postproc' ! fakesink
```
New property `segments` take a string that is a comma-separated key=value pairs. Key represents a segment name that will be used in downstream element, and value is the path to the file that represents a binary data associated with the given segment

NOTE: User shall use **only one** of these two properies: `segments` or `location`.

## Configuration

No configuration needed for this plugin

## Contributing

Submit you PRs to https://bitbucket.org/sima-ai/gst-simaai-plugins-base/src/develop/gst/simaaisrc/

## License

## Acknowledgments
