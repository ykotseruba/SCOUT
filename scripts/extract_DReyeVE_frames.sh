#!/bin/bash
# extract_DREYEVE_frames.sh [etg | garmin]

camera=$1 


for video_dir in ${DREYEVE_PATH}/*
do
    if [ -d $video_dir ]
    then
    	echo $video_dir
        video_file=$video_dir/video_${camera}.avi
        if [ $camera == etg ]
       	then
        	frame_dir=$video_dir/frames_${camera}
        else
        	frame_dir=$video_dir/frames
        fi
        mkdir -p $frame_dir
        rm -rf ${frame_dir}/*jpg
        rm -rf ${frame_dir}/*png

        #FFMPEG will overwrite any existing images in that directory
        ffmpeg  -y -i $video_file -start_number 0 -f image2 -q:v 0 ${frame_dir}/%06d.jpg
    fi
done


