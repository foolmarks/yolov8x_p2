#!/bin/sh

# preproc dependent application
sh /data/simaai/applications/resnet/etc/load_ev_cfg_preproc.sh

#postproc dependent application
sh /data/simaai/applications/resnet/etc/load_ev_cfg_postproc.sh

#run the gst_app with the necessary config
GST_DEBUG=3 LD_LIBRARY_PATH=./lib GST_PLUGIN_PATH=./lib ./build/gst_app  --gst-string="simaaisrc location=\"/data/simaai/applications/resnet/etc/img_1_1920_1080.rgb\" node-name=\"input\" blocksize=6220800 delay=1000 mem-target=1 ! process-cvu source-node-name=\"input\" buffers-list=\"input\" transmit=true config=\"/data/simaai/applications/resnet/etc/simaai_detess_dequant_preproc_1.json\"  !  process2 transmit=true config=\"/data/simaai/applications/resnet/etc/process_mla_1.json\"  !  process-cvu source-node-name=\"process_mla\" buffers-list=\"process_mla\" transmit=true config=\"/data/simaai/applications/resnet/etc/simaai_dequantize_postproc_1.json\" ! fakesink" --manifest-json="/mnt/aseem/resnet/manifest.json"
