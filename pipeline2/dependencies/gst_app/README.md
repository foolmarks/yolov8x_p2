# README #

### What is this repository for? ###

The repository contains the template code for a c++ based gstreamer application.

### How do I get set up? ###

Steps to build the gst-app:
1. Clone the repo: `git clone git@bitbucket.org:sima-ai/vdp-gst-application.git`    
2. `cd vdp-gst-application`  
3. Checkout the develop branch: `git checkout develop`     
4. `mkdir build && cd build`  
5. `cmake ..`  
6. `make` This will create the `gst-app` executable.   

To run the gst-app:  
```
Usage: ./build/gst_app [options]
Options:
  --manifest-json <path>  Path to the manifest JSON file (required)
  --gst-string <string>   GStreamer pipeline string (required)
  --rtsp-url <urls>       "url1 url2 url3" Space-separated list of RTSP URLs (optional)
  --host-ip <ips>         "ip1 ip2 ip3" Space-separated list of host IP addresses (optional)
  --host-port <ports>     "port1 port2 port3" Space-separated list of host port numbers (optional)  
  --gst_string_replacements <jsonStr> Gst string replacement json string (optional)
```

Gst string repalcement Json format:  
```
{
  "TAG_1" : "REPLACEMENT_1",
  "TAG_2" : "REPLACEMENT_2",
  ...
}
```

Please note that all necessary dependencies:  
1. `GST_PLUGIN_PATH`    
2. `LD_LIBRARY_PATH`    
3. CVU Dependent applications     
Must be run before before the gst-app  


### Pipelines Currently Supported (All Ethernet pipelines) ###  
1. ResNet 50   
2. YoloV5  
3. YoloV7  
4. People Detector  
5. People Tracker  
6. DETR  
7. YoloV8    
8. Openpose  
9. Face Mask Detector  
10. EfficientDet  
  
### Who do I talk to? ###
  
Repo Owned by: Aseem Kannal