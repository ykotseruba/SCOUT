#!/bin/bash
# extract_BDDA_frames.sh


#for set_dir in train validation test
for set_dir in training validation test
do
    for video in ${BDDA_PATH}/${set_dir}/camera_videos/*.mp4
    do
        filename=$(basename "$video")
        fname="${filename%.*}"

        #create a directory for each frame sequence
        
        #extract camera frames
        save_dir=$BDDA_PATH/${set_dir}/camera_frames/$fname
        mkdir -p $save_dir
        #FFMPEG will overwrite any existing images in that directory
        ffmpeg  -y -i $video -start_number 0 -f image2 -q:v 2 -r 29 ${save_dir}/%05d.jpg
        
        #extract ground truth
        save_dir=$BDDA_PATH/${set_dir}/gazemap_frames/$fname
        mkdir -p $save_dir
        # BDD-A readme says that central region of 1024x576 of gazemap video corresponds to the frame of the camera video
        # add filter option to ffmpeg to crop black bars on the top and bottom of the original frame 1024x768
        ffmpeg  -y -i $BDDA_PATH/$set_dir/gazemap_videos/${fname}_pure_hm.mp4 -filter:v "crop=1024:576:0:91" -start_number 0 -f image2 -q:v 2 -r 29 ${save_dir}/%05d.png
    done
done



# this script verifies that the number of frames in camera and ground truth matches
# outputs videos with different frame counts

for set_dir in training validation test
do
    for video in ${BDDA_PATH}/$set_dir/camera_videos/*.mp4
    do
        filename=$(basename "$video")
        fname="${filename%.*}"
        
        gaze_frame_count=$(ls $BDDA_PATH/$set_dir/gazemap_frames/$fname | wc -l)
        camera_frame_count=$(ls $BDDA_PATH/$set_dir/camera_frames/$fname | wc -l)

        if [ $gaze_frame_count -ne $camera_frame_count ]
        then
            echo $set_dir $fname gaze_count=$gaze_frame_count camera_count=$camera_frame_count
        fi
    done
done