# YoloV8x_p2 Object Detection #

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
> The pre-processing used in this Yolov8x_p2 example is conversion from BGR to RGB format, resizing and padding to match the input dimensions while maintaing the correct aspect ratio, there is no normalization, means subtraction, etc.


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

Images annotated with bounding boxes are writted itno the ./build/onnx_pred folder.


<img src="./readme_images/onnx_test_000000574520.jpg" alt="" style="height: 500px; width:500px;"/>



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
--------------------------------------------------
Model SDK version 1.7.0
3.10.12 (main, Aug  6 2025, 18:09:36) [GCC 11.4.0]
--------------------------------------------------
Removing existing directory: /home/docker/sima-cli/build/yolox_s_opt_no_reshapes
Results will be written to /home/docker/sima-cli/build/yolox_s_opt_no_reshapes
--------------------------------------------------
Model Inputs
images: (1, 3, 640, 640)
--------------------------------------------------
Loaded model from yolox_s_opt_no_reshapes.onnx
Quantizing with 50 calibration samples
Calibration Progress: |██████████████████████████████| 100.0% 50|50 Complete.  50/50
Running Calibration ...DONE
Running quantization ...DONE
Quantized model saved to /home/docker/sima-cli/build/yolox_s_opt_no_reshapes/yolox_s_opt_no_reshapes.sima.json
Removing existing directory: /home/docker/sima-cli/build/quant_pred
Annotated images will be written to /home/docker/sima-cli/build/quant_pred
Processing image: /home/docker/sima-cli/test_images/000000006894.jpg
Wrote output image to /home/docker/sima-cli/build/quant_pred/test_000000006894.jpg
Processing image: /home/docker/sima-cli/test_images/000000019221.jpg
Wrote output image to /home/docker/sima-cli/build/quant_pred/test_000000019221.jpg
Processing image: /home/docker/sima-cli/test_images/000000022589.jpg
Wrote output image to /home/docker/sima-cli/build/quant_pred/test_000000022589.jpg
Processing image: /home/docker/sima-cli/test_images/000000032941.jpg
Wrote output image to /home/docker/sima-cli/build/quant_pred/test_000000032941.jpg
Processing image: /home/docker/sima-cli/test_images/000000048504.jpg
Wrote output image to /home/docker/sima-cli/build/quant_pred/test_000000048504.jpg
Processing image: /home/docker/sima-cli/test_images/000000572408.jpg
Wrote output image to /home/docker/sima-cli/build/quant_pred/test_000000572408.jpg
Processing image: /home/docker/sima-cli/test_images/000000573626.jpg
Wrote output image to /home/docker/sima-cli/build/quant_pred/test_000000573626.jpg
Processing image: /home/docker/sima-cli/test_images/000000574520.jpg
Wrote output image to /home/docker/sima-cli/build/quant_pred/test_000000574520.jpg
Processing image: /home/docker/sima-cli/test_images/000000577735.jpg
Wrote output image to /home/docker/sima-cli/build/quant_pred/test_000000577735.jpg
Processing image: /home/docker/sima-cli/test_images/000000581062.jpg
Wrote output image to /home/docker/sima-cli/build/quant_pred/test_000000581062.jpg
Compiling with batch size set to 1
Wrote compiled model to /home/docker/sima-cli/build/yolox_s_opt_no_reshapes/yolox_s_opt_no_reshapes_mpk.tar.gz
```



The evaluation of the quantized model generates images annotated with bounding boxes and are written into the ./build/quant_pred folder.


<img src="./readme_images/quant_test_000000574520.jpg" alt="" style="height: 500px; width:500px;"/>



### Code Walkthrough for run_modelsdk.py ###


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
Removing existing directory: /home/docker/sima-cli/build/accel_pred
Annotated images will be written to /home/docker/sima-cli/build/accel_pred
Loading yolox_s_opt_no_reshapes quantized model from build/yolox_s_opt_no_reshapes
Using 10 out of 10  test images
Compiling model yolox_s_opt_no_reshapes to .elf file
Creating the Forwarding from host
Copying the model files to DevKit
Creating the Forwarding from host
ZMQ Connection successful.
Executing model graph in accelerator mode:
Progress: |██████████████████████████████| 100.0% 10|10 Complete.  10/10
Model is executed in accelerator mode.
Processing image: /home/docker/sima-cli/test_images/000000006894.jpg
Wrote output image to /home/docker/sima-cli/build/accel_pred/test_000000006894.jpg
Processing image: /home/docker/sima-cli/test_images/000000019221.jpg
Wrote output image to /home/docker/sima-cli/build/accel_pred/test_000000019221.jpg
Processing image: /home/docker/sima-cli/test_images/000000022589.jpg
Wrote output image to /home/docker/sima-cli/build/accel_pred/test_000000022589.jpg
Processing image: /home/docker/sima-cli/test_images/000000032941.jpg
Wrote output image to /home/docker/sima-cli/build/accel_pred/test_000000032941.jpg
Processing image: /home/docker/sima-cli/test_images/000000048504.jpg
Wrote output image to /home/docker/sima-cli/build/accel_pred/test_000000048504.jpg
Processing image: /home/docker/sima-cli/test_images/000000572408.jpg
Wrote output image to /home/docker/sima-cli/build/accel_pred/test_000000572408.jpg
Processing image: /home/docker/sima-cli/test_images/000000573626.jpg
Wrote output image to /home/docker/sima-cli/build/accel_pred/test_000000573626.jpg
Processing image: /home/docker/sima-cli/test_images/000000574520.jpg
Wrote output image to /home/docker/sima-cli/build/accel_pred/test_000000574520.jpg
Processing image: /home/docker/sima-cli/test_images/000000577735.jpg
Wrote output image to /home/docker/sima-cli/build/accel_pred/test_000000577735.jpg
Processing image: /home/docker/sima-cli/test_images/000000581062.jpg
Wrote output image to /home/docker/sima-cli/build/accel_pred/test_000000581062.jpg
```


The evaluation of the compiled model generates images annotated with bounding boxes and are written into the ./build/accel_pred folder.



<img src="./readme_images/accel_test_000000574520.jpg" alt="" style="height: 500px; width:500px;"/>




## Benchmarking model on hardware ##

The model can be benchmarked on the target board. This uses random data to test the throughput - note that this only tests the MLA throughput.



```shell
python ./get_fps/network_eval/network_eval.py \
    --model_file_path   ./build/yolox_s_opt_no_reshapes/benchmark/yolox_s_opt_no_reshapes_stage1_mla.elf \
    --mpk_json_path     ./build/yolox_s_opt_no_reshapes/benchmark/yolox_s_opt_no_reshapes_mpk.json \
    --dv_host           192.168.8.20 \
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
sima@192.168.1.29's password: 
FPS = 560
FPS = 566
FPS = 568
FPS = 569
FPS = 570
FPS = 571
FPS = 571
FPS = 571
FPS = 572
FPS = 572
Ran 100 frame(s)
```

Note that this is the throughput of only the MLA (i.e. the YoloX model), it does not include any pre or post-processing.


## Building the GStreamer Pipeline ##


## Files & Folders

* yolo_s.onnx -  original YoloX Small model (not used)
* yolox_surgery_no_reshape.py - graph surgery script (not used)
* yolo_s_opt_no_reshapes.onnx - post surgery ONNX model
* run_onnx.py - executes and evaluates the floating-point ONNX model
* run_modelsdk.py - quantizes & compiles, optionally evaluates the quantized model
* run_accelmode.py - executes the model in hardware
* utils.py - common utilities functions
* payload_contents.py - contents of custom Python plugin
* yolox_s_opt_no_reshapes_mpk_rtspsrc - working version of the final GStreamer pipeline
* get_fps - scripts for benchmarking
* calib_images.zip - images for calibration
* test_images.zip - images for testing

## Acknowledgements

* 'davsolai' on HuggingFace for the trained PyTorch model


