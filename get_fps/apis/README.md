### This file describes how to run the PePPi pipeline with remote runner.

#### Step 1: Copy the SDK Contents to Your Host System
Use the following command to copy the contents of this example to your host system from the SDK.

$ cp -r /usr/local/simaai/app_zoo/model-accelerator-mode/ /home/docker/sima-cli/

#### Step2: Execute the Examples
On your host system, browse to the shared folder where you copied the example contents and follow these steps to execute the examples.

#### Prepare Host to run remote runner
```bash
    $ cd devkit-inference-examples/apis
    
    $ python3 -m venv test_venv
    $ source test_venv/bin/activate
    $ pip3 install -r requirements.txt
```

### Example: Centernet

#### Install GStreamer on the host

```
    brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-ffmpeg
    gst-launch-1.0 --version
    gst-launch-1.0 version 1.22.3
    GStreamer 1.22.3
    https://github.com/Homebrew/homebrew-core
```

### Package up files.

Create a tarball `test_centernet.tar.gz` containing `UR_onnx_people-centernet_fp32_512_512_mpk.tar.gz`,
`get_top_K_accuracy_objects.py`, `utils.py`, and `test_centernet.py`.

```bash
    $ tar -czvf test_centernet.tar.gz *
```

#### Run the test script from the host.

```bash
    $ cd devkit-inference-examples/apis
    
    $ python remote_runner.py --dv_host {host name} --model_file_path test_centernet.tar.gz --model_command 'test_centernet.py --network_src 5000 --model_tgz UR_onnx_people-centernet_fp32_512_512_mpk.tar.gz --image_width 1280 --image_height 720 --host_IP {IP_address} --gst_port 8000' --run_time 60

```
>> change `run_time` as necessary as per the video or the test requirement


#### Feed the data from the host.

```bash
    $ gst-launch-1.0 filesrc location=people_1280x720.mp4 ! decodebin ! videoconvert ! video/x-raw,format=NV12 ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=dm17.sjc.sima.ai  port=5000
```

#### Stream the video on the host
```bash
 GST_DEBUG=0 gst-launch-1.0 udpsrc port=8000 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! 'video/x-h264,stream-format=byte-stream,alignment=au' !  avdec_h264  ! queue2 ! queue max-size-bytes=15728640 ! autovideosink sync=0
```

### Example: Resnet
### Package up files.

Create a tarball `test_resnet.tar.gz` containing `resnet50_v1_opt_mpk.tar.gz` and `test_resnet50_v1.py`.

```bash
    $ tar -czvf test_resnet.tar.gz *
```
Copy the imagenet files and lable file to the board
scp -r /data/resnet50_demo sima@<board>.sjc.sima.ai:/home/sima
scp imagenet1000_clsidx_to_labels.txt sima@<board>.sjc.sima.ai:/home/sima

#### Run the test script from the host.

```bash
    $ cd devkit-inference-examples/apis
    
    $ python remote_runner.py --dv_host dm6.sjc.sima.ai --model_file_path test_resnet.tar.gz --model_command 'test_resnet50_v1.py  --model_tgz resnet50_v1_opt_mpk.tar.gz --image_width 1280 --image_height 720 --max_frames 10 --images_path /home/sima --images_labels_path /home/sima' --run_time 60

```
>> change `run_time` as necessary as per the video or the test requirement
