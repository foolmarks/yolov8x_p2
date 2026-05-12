# NEAT Python application task specification


## working envornment
* ghcr.io-sima-neat-sdk-latest docker container

## task
Create a Python NEAT application named run_neat_sync.py that does the following:

* use synchronous execution
* read all images from 'test_images' folder - use get_image_paths() from utils.py
* push the images into the pipeline
* pull the output when ready
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
* target folder: /home/mark/projects/tattile/yolov8x_p2/neat/py
    * make the target folder if it does not exist
* results folder: /home/mark/projects/tattile/yolov8x_p2/neat/py/results
* compiled model: /home/mark/projects/tattile/yolov8x_p2/build/yolov8x-p2_opt_4o/yolov8x-p2_opt_4o_mpk.tar.gz
* images: /home/mark/projects/tattile/yolov8x_p2/test_images
* utils module: /home/mark/projects/tattile/yolov8x_p2/utils.py

