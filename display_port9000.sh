gst-launch-1.0 udpsrc port=9000 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! 'video/x-h264,stream-format=byte-stream,alignment=au' !  avdec_h264  ! fpsdisplaysink sync=0 -v
#GST_DEBUG=0 gst-launch-1.0 udpsrc port=9000 ! 'application/x-rtp,encoding-name=H264,clock-rate=90000,payload=96' ! rtph264depay ! h264parse ! 'video/x-h264,stream-format=byte-stream,alignment=au' !  avdec_h264  ! queue2 max-size-bytes=15728640 ! autovideosink


