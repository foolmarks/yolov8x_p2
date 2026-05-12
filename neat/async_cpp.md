# NEAT C++ application task specification
Prompt for codex/claude.


## working envornment
* ghcr.io-sima-neat-sdk-latest docker container


## task
Create a C++ NEAT application named main_async.cpp that does the following:

* use asynchronous execution
* read all images from 'test_images' folder - use get_image_paths() from utils.py as a guide.
* push the images into the pipeline, pull the output when ready
    * push operations and pull operations should be seperate.
    * use run.try_push() to push without blocking.
    * check run.can_push() and run.running() after building the pipeline.
    * do not use push_and_pull(), model.run(), session.run(), or other synchronous convenience operations.
    * do not call run.close_input() immediately after pushing the last image; drain the expected outputs first, then close the run.
    * bound the number of in-flight images so outputs can be drained while new inputs are admitted.
    * if try_push() returns false, pull pending output and retry later instead of blocking the producer.
* run post-processing and image write (see post-processing below)



### image-preprocessing
* execute on CVU - use generic preproc kernel
* functionality required:
    * letterbox resize & pad to 1, 640, 640, 3, maintaining aspect ratio.
    * normalization to range 0, 1.0
    * quantization
* images must be passed to model in RGB layout
* refer to preprocess_image() function in utils.py as a guide.


### model
* contained in compiled model tar.gz archive (see paths below)
* trained on COCO dataset
* input dimensions are NHWC = 1,640,640,3



### post-processing
* box decoding, NMS
    * refer to postprocess_yolov8x_p2_4o() function in utils.py as a guide.
* draw detections on 640 x 640 image that was input to model
    * refer to draw_detections() function in utils.py as a guide.
* Write images with overlays into a folder named 'results' - refer to prepare_output_dir() from utils.py as a guide.
    * Delete and recreate 'results' if it already exists - refer to prepare_output_dir() in utils.py as a guide.



## testing 
* test and verify on target board at IP address 192.168.1.20


## skills
* Use SiMa NEAT skill



## paths
* target folder: /home/mark/projects/tattile/yolov8x_p2/neat/cpp
    * make the target folder if it does not exist
* results folder: /home/mark/projects/tattile/yolov8x_p2/neat/cpp/results
* compiled model: /home/mark/projects/tattile/yolov8x_p2/build/yolov8x-p2_opt_4o/yolov8x-p2_opt_4o_mpk.tar.gz
* images: /home/mark/projects/tattile/yolov8x_p2/test_images
* utils module: /home/mark/projects/tattile/yolov8x_p2/utils.py

