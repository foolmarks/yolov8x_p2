# YoloV8x_p2 Object Detection #

# WORK IN PROGRESS! #

This tutorial demonstrates the following steps:

  * Running a trained FP32 ONNX model to provide baseline results.
  * Loading, Quantizing and Compiling of the trained FP32 model.
  * Evaluation of the quantized model.
  * Benchmarking of the compiled model.
  * Executing a compiled model on a hardware platform.
  * Building a GStreamer pipeline with IP camera input


YOLOv8x uses three detection heads on feature maps P3, P4, P5, corresponding to strides 8, 16 and 32 (so for a 640×640 input you get 80×80, 40×40, 20×20 grids). The p2 variant modifies the architecture so that the detector outputs from P2–P5 instead of P3–P5:

* P2 head (new) – stride 4, feature map 160×160 for 640×640 input
* P3 head – stride 8 (80×80)
* P4 head – stride 16 (40×40)
* P5 head – stride 32 (20×20)

Architecturally this is done by extending the neck: additional upsampling and concat operations pull higher-resolution features (P2) from the backbone, process them with C2f blocks, and then feed a 4-scale Detect head: Detect(P2, P3, P4, P5). The P2 head operates on a higher-resolution feature map (stride 4), so bounding boxes for tiny objects (e.g. 5–15 px in one dimension) can be anchored to more precise cell centers.

The p2 variant is suited to high-resolution, cluttered scenes or large images with many tiny objects (UAV imagery, drones, traffic signs at distance, insects, cells, PCB components, etc.). 


The disadvantages are generally slower inference and higher memory usage. Also, the labelling of small objects can sometimes be error-prone, if the small-object annotations in the training dataset are noisy or inconsistent, the P2 head will overfit that noise. which leads to higher false positives on cluttered backgrounds unless you tune augmentations and losses carefully.



## Starting the Palette SDK docker container ##

The docker container can be started by running the start.py script from the command line:

```shell
python start.py
```
When asked to enter the the work directory, just respond with `./`


The output in the console should look something like this:

```shell
/home/projects/modelsdk_accelmode/./start.py:111: SyntaxWarning: invalid escape sequence '\P'
  docker_start_cmd = 'cmd.exe /c "start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe""'
Set no_proxy to localhost,127.0.0.0
Using port 49152 for the installation.
Checking if the container is already running...
Enter work directory [/home/projects/modelsdk_accelmode]: ./
Starting the container: palettesdk_1_7_0_Palette_SDK_master_B219
Checking SiMa SDK Bridge Network...
SiMa SDK Bridge Network found.
Creating and starting the Docker container...
f72ae89b3c12494291a4b9f621f8d28f565705b8dd638fc058a79bfb5ce5e73c
Successfully copied 3.07kB to /home/projects/modelsdk_accelmode/passwd.txt
Successfully copied 3.07kB to palettesdk_1_7_0_Palette_SDK_master_B219:/etc/passwd
Successfully copied 2.56kB to /home/projects/modelsdk_accelmode/shadow.txt
Successfully copied 2.56kB to palettesdk_1_7_0_Palette_SDK_master_B219:/etc/shadow
Successfully copied 2.56kB to /home/projects/modelsdk_accelmode/group.txt
Successfully copied 2.56kB to palettesdk_1_7_0_Palette_SDK_master_B219:/etc/group
Successfully copied 3.58kB to /home/projects/modelsdk_accelmode/sudoers.txt
Successfully copied 3.58kB to palettesdk_1_7_0_Palette_SDK_master_B219:/etc/sudoers
Successfully copied 2.05kB to palettesdk_1_7_0_Palette_SDK_master_B219:/home/docker/.simaai/.port
user@f72ae89b3c12:/home$
```

Navigate into the working directory:

```shell
cd docker/sima-cli
```

Unzip the test and calibration images:

```shell
unzip calib_images.zip
unzip test_images.zip
```


## Converting the PyTorch model to ONNX ##

The starting point will be a trained PyTorch model - the model file (model.pt) is provided from [Google Drive](https://drive.google.com/file/d/1xXnEIUwK4isMbNv6ZDlpC614Om6v7ota/view?usp=sharing) or can be downloaded from its original [source repository](https://huggingface.co/davsolai/yolov8x-p2-coco).

First, we will convert it to ONNX format:

```shell
python export2onnx.py
```



## Execute The Original Floating-Point ONNX model ##

ONNXRuntime is included in the SDK docker, so we can run the floating-point model. This is useful to provide a baseline for comparing to the post-quantization and post-compile models. The run_onnx.py script includes pre- and postprocessing.

> Note: The same pre-processing and post-processing is used at every step and will generally be similar to the pre/post-processing used during model training.
> The pre-processing used in this Yolov8x_p2 example is resizing and padding to the model input size (usually 640,64) conversion from BGR to RGB format, pixel value normalization to the range 0->1s


```shell
python run_onnx_single_out.py 
```


The expected console output is like this:

```shell
user@7df5555533b9:/home/docker/sima-cli$ python run_onnx_single_out.py 

--------------------------------------------------
3.10.12 (main, Aug  6 2025, 18:09:36) [GCC 11.4.0]
--------------------------------------------------
Found 10 image(s) in './test_images'
Output images will be written to './build/onnx_pred'
Processing image: 000000006894.jpg
  Detections: 2
  Annotated image written to: ./build/onnx_pred/000000006894.jpg
Processing image: 000000019221.jpg
  Detections: 2
  Annotated image written to: ./build/onnx_pred/000000019221.jpg
Processing image: 000000022589.jpg
  Detections: 2
  Annotated image written to: ./build/onnx_pred/000000022589.jpg
Processing image: 000000032941.jpg
  Detections: 12
  Annotated image written to: ./build/onnx_pred/000000032941.jpg
Processing image: 000000048504.jpg
  Detections: 4
  Annotated image written to: ./build/onnx_pred/000000048504.jpg
Processing image: 000000572408.jpg
  Detections: 4
  Annotated image written to: ./build/onnx_pred/000000572408.jpg
Processing image: 000000573626.jpg
  Detections: 1
  Annotated image written to: ./build/onnx_pred/000000573626.jpg
Processing image: 000000574520.jpg
  Detections: 2
  Annotated image written to: ./build/onnx_pred/000000574520.jpg
Processing image: 000000577735.jpg
  Detections: 2
  Annotated image written to: ./build/onnx_pred/000000577735.jpg
Processing image: 000000581062.jpg
  Detections: 2
  Annotated image written to: ./build/onnx_pred/000000581062.jpg
```

Images annotated with bounding boxes are written into the ./build/onnx_pred folder.


<img src="./readme_images/onnx_1_000000022589.jpg" alt="" style="height: 400px; width:500px;"/>



## Graph Surgery ##

```shell
python rewrite_yolov8mp2_4_outs.py
```

## Execute The Modified Floating-Point ONNX model ##

```shell
python run_onnx_4_outs.py
```


Images annotated with bounding boxes are written into the ./build/onnx_4_pred folder.


<img src="./readme_images/onnx_4_000000022589.jpg" alt="" style="height: 400px; width:500px;"/>




## Quantize & Compile ##

The run_modelsdk.py script will do the following:

* load the floating-point ONNX model.
* quantize using pre-processed calibration data and quantization parameters set using command line arguments.
* test the quantized model accuracy using pre-processed images. Annotated images are written to build/quant_pred
* compile and then untar to extract the .elf and .json files (for use in benchmarking on the target board)

*Note: the quantization is done using min-max calibration method instead of the default mse method.*


```shell
python run_modelsdk.py -e
```

The images are written into build/quant_pred folder:


The expected console output is like this:

```shell
mark@7df5555533b9:/home/docker/sima-cli$ python run_modelsdk.py -e

--------------------------------------------------
Model SDK version 1.7.0
3.10.12 (main, Aug  6 2025, 18:09:36) [GCC 11.4.0]
--------------------------------------------------
Results will be written to /home/docker/sima-cli/build/yolov8x-p2_opt_4o
--------------------------------------------------
Model Inputs:
images: (1, 3, 640, 640)
--------------------------------------------------
2025-11-19 03:04:09,940 - afe.apis.loaded_net - INFO - Loading ['yolov8x-p2_opt_4o.onnx'] in onnx format
Loaded model from yolov8x-p2_opt_4o.onnx
Quantizing with 1 calibration samples
2025-11-19 03:04:21,165 - afe.apis.loaded_net - INFO - Quantize loaded net, layout = NCHW, arm_only = False
2025-11-19 03:04:21,165 - afe.apis.loaded_net - INFO - Calibration method = min_max
2025-11-19 03:05:59,219 - afe.ir.transform.calibration_transforms - INFO - Running Calibration ...
Calibration Progress: |██████████████████████████████| 100.0% 1|1 Complete.  1/1
Running Calibration ...DONE
Running quantization ...DONE
Quantized model saved to /home/docker/sima-cli/build/yolov8x-p2_opt_4o/yolov8x-p2_opt_4o.sima.json
Annotated images will be written to /home/docker/sima-cli/build/quant_pred
Using 10 out of 10  test images
Processing image: 000000006894.jpg
  Detections: 2
  Annotated image written to: /home/docker/sima-cli/build/quant_pred/000000006894.jpg
Processing image: 000000019221.jpg
  Detections: 2
  Annotated image written to: /home/docker/sima-cli/build/quant_pred/000000019221.jpg
Processing image: 000000022589.jpg
  Detections: 2
  Annotated image written to: /home/docker/sima-cli/build/quant_pred/000000022589.jpg
Processing image: 000000032941.jpg
  Detections: 12
  Annotated image written to: /home/docker/sima-cli/build/quant_pred/000000032941.jpg
Processing image: 000000048504.jpg
  Detections: 4
  Annotated image written to: /home/docker/sima-cli/build/quant_pred/000000048504.jpg
Processing image: 000000572408.jpg
  Detections: 4
  Annotated image written to: /home/docker/sima-cli/build/quant_pred/000000572408.jpg
Processing image: 000000573626.jpg
  Detections: 1
  Annotated image written to: /home/docker/sima-cli/build/quant_pred/000000573626.jpg
Processing image: 000000574520.jpg
  Detections: 2
  Annotated image written to: /home/docker/sima-cli/build/quant_pred/000000574520.jpg
Processing image: 000000577735.jpg
  Detections: 2
  Annotated image written to: /home/docker/sima-cli/build/quant_pred/000000577735.jpg
Processing image: 000000581062.jpg
  Detections: 2
  Annotated image written to: /home/docker/sima-cli/build/quant_pred/000000581062.jpg
Compiling with batch size set to 1
Wrote compiled model to /home/docker/sima-cli/build/yolov8x-p2_opt_4o/yolov8x-p2_opt_4o_mpk.tar.gz
```



The evaluation of the quantized model generates images annotated with bounding boxes and are written into the ./build/quant_pred folder.


<img src="./readme_images/quant_pred_000000022589.jpg" alt="" style="height: 400px; width:500px;"/>





## Test model on hardware ##

Run the model directly on the target board. This requires the target board to be reachable via ssh. Make sure to set the IP address of the target board:


```shell
python run_accelmode.py -hn <target_ip_address>
```



The output in the console will be something like this:


```shell
user@c88084ef7e7b:/home/docker/sima-cli$ python run_accelmode.py 

--------------------------------------------------
Model SDK version 1.7.0
3.10.12 (main, Aug  6 2025, 18:09:36) [GCC 11.4.0]
--------------------------------------------------
Annotated images will be written to /home/docker/sima-cli/build/accel_pred
Loading yolov8x-p2_opt_4o quantized model from build/yolov8x-p2_opt_4o
Using 10 out of 10  test images
Processing image: 000000006894.jpg
Processing image: 000000019221.jpg
Processing image: 000000022589.jpg
Processing image: 000000032941.jpg
Processing image: 000000048504.jpg
Processing image: 000000572408.jpg
Processing image: 000000573626.jpg
Processing image: 000000574520.jpg
Processing image: 000000577735.jpg
Processing image: 000000581062.jpg
Compiling model yolov8x-p2_opt_4o to .elf file
Creating the Forwarding from host
Copying the model files to DevKit
Creating the Forwarding from host
ZMQ Connection successful.
Executing model graph in accelerator mode:
Progress: |██████████████████████████████| 100.0% 10|10 Complete.  10/10
Model is executed in accelerator mode.
  Detections: 2
  Annotated image written to: /home/docker/sima-cli/build/accel_pred/000000006894.jpg
  Detections: 2
  Annotated image written to: /home/docker/sima-cli/build/accel_pred/000000019221.jpg
  Detections: 2
  Annotated image written to: /home/docker/sima-cli/build/accel_pred/000000022589.jpg
  Detections: 12
  Annotated image written to: /home/docker/sima-cli/build/accel_pred/000000032941.jpg
  Detections: 4
  Annotated image written to: /home/docker/sima-cli/build/accel_pred/000000048504.jpg
  Detections: 4
  Annotated image written to: /home/docker/sima-cli/build/accel_pred/000000572408.jpg
  Detections: 1
  Annotated image written to: /home/docker/sima-cli/build/accel_pred/000000573626.jpg
  Detections: 2
  Annotated image written to: /home/docker/sima-cli/build/accel_pred/000000574520.jpg
  Detections: 2
  Annotated image written to: /home/docker/sima-cli/build/accel_pred/000000577735.jpg
  Detections: 2
  Annotated image written to: /home/docker/sima-cli/build/accel_pred/000000581062.jpg
```


The evaluation of the compiled model generates images annotated with bounding boxes and are written into the ./build/accel_pred folder.



<img src="./readme_images/accel_pred_000000022589.jpg" alt="" style="height: 400px; width:500px;"/>




## Benchmarking model on hardware ##

The model can be benchmarked on the target board. This uses random data to test the throughput - note that this only tests the MLA throughput.



```shell
python ./get_fps/network_eval/network_eval.py \
    --model_file_path   ./build/yolov8x-p2_opt_4o/benchmark/yolov8x-p2_opt_4o_stage1_mla.elf \
    --mpk_json_path     ./build/yolov8x-p2_opt_4o/benchmark/yolov8x-p2_opt_4o_mpk.json \
    --dv_host           <target_ip_address> \
    --image_size        640 640 3 \
    --verbose \
    --bypass_tunnel \
    --max_frames        100 \
    --batch_size        1
```

  The measured throughput in frames per second (FPS) will be printed in the console:

```shell
Running model in MLA-only mode
Copying the model files to DevKit
sima@192.168.1.21's password: 
FPS = 108
FPS = 109
FPS = 109
FPS = 109
FPS = 109
FPS = 109
FPS = 109
FPS = 109
FPS = 109
FPS = 109
Ran 100 frame(s)
```

Note that this is the throughput of only the MLA (i.e. the modified YoloV8x_p2 model), it does not include any pre or post-processing.


## Building the GStreamer Pipeline ##

Make sample images that can be used with the simaaisrc plugin:

```shell
python make_samples_640.py
```


Make the baseline pipeline:

```shell
mpk project create --model-path ./build/yolov8x-p2_opt_4o/yolov8x-p2_opt_4o_mpk.tar.gz --src-plugin simaaisrc --input-resource ./build/samples_640/img%d.rgb --input-width 640 --input-height 640 --input-img-type RGB
```


### Add the Python custom plugin ###


Modify .project/pluginsInfo.json to add a new plugin called 'yolov8xp2_postproc_overlay' - this will be the   :


```json
{
    "pluginsInfo": [
        {
            "gid": "processcvu",
            "path": "plugins/processcvu"
        },
        {
            "gid": "processmla",
            "path": "plugins/processmla"
        },
        {
            "gid" : "yolov8xp2_postproc_overlay",
            "path" : "plugins/yolov8xp2_postproc_overlay"
        }
    ]
}
```



Add a folder to contain the custom Python plugin and copy the templates into it:

```shell
mkdir -p ./yolov8x-p2_opt_4o_mpk_simaaisrc/plugins/yolov8xp2_postproc_overlay
cp /usr/local/simaai/plugin_zoo/gst-simaai-plugins-base/gst/templates/aggregator_python/python/*.py ./yolov8x-p2_opt_4o_mpk_simaaisrc/plugins/yolov8xp2_postproc_overlay/.
```


Open 'application.json' and add the following into the "plugins" section of the JSON file:


```shell
    {
      "name": "simaai_yolov8xp2_postproc_overlay",
      "pluginGid": "yolov8xp2_postproc_overlay",
      "sequence" : 5
    }
```


Modify the 'gst' string:


```shell
    "gst": "simaaisrc location=/data/simaai/applications/yolov8x-p2_opt_4o_mpk_simaaisrc/etc/img%d.rgb node-name=decoder delay=1000 mem-target=1 index=1 loop=true ! 'video/x-raw, format=(string)RGB, width=(int)640, height=(int)640' ! tee name=source ! queue2 ! simaaiprocesscvu  name=simaaiprocesspreproc_1 ! simaaiprocessmla  name=simaaiprocessmla_1 ! simaaiprocesscvu  name=simaaiprocessdetess_dequant_1 ! yolov8xp2_postproc_overlay  name='simaai_yolov8xp2_postproc_overlay' ! queue2 ! 'video/x-raw, format=(string)RGB, width=(int)640, height=(int)640' ! fakesink source. ! queue2 ! simaai_yolov8xp2_postproc_overlay. "
```







Compile the pipeline:

```shell
mpk create --clean --board-type modalix -d ./yolov8x-p2_opt_4o_mpk_simaaisrc -s ./yolov8x-p2_opt_4o_mpk_simaaisrc
```



Deploy

```shell
mpk device connect -d devkit -u sima -p edgeai -t 192.168.1.21
mpk deploy -f ./yolov8x-p2_opt_4o_mpk_simaaisrc/project.mpk -d devkit -t 192.168.1.21
```






## Files & Folders

* rewrite_yolov8mp2_4_outs.py - graph surgery script
* yolo_model.py - graph surgery utilities
* run_onnx_single_output.py - executes and evaluates the single output floating-point ONNX model
* run_onnx_4_outs.py - executes and evaluates the post-surgery floating-point ONNX model
* run_modelsdk.py - quantizes & compiles, optionally evaluates the quantized model
* run_accelmode.py - executes the model in hardware
* utils.py - common utilities functions
* get_fps - scripts for benchmarking
* calib_images.zip - images for calibration
* test_images.zip - images for testing
* start.py - start docker container
* stop.py stop docker container

## Acknowledgements

* 'davsolai' on HuggingFace for the trained PyTorch model


