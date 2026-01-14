gst-launch-1.0 udpsrc port=9000 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! 'video/x-h264,stream-format=byte-stream,alignment=au' !  avdec_h264  ! fpsdisplaysink sync=0 -v

#gst-launch-1.0 udpsrc port=9000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" !   rtph264depay ! h264parse ! avdec_h264 ! videoconvert !   fpsdisplaysink video-sink=autovideosink text-overlay=true sync=false

