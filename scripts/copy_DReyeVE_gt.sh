#!/bin/bash


for vid in {1..74}; do 
vid_id=$(printf %02d $vid)
mkdir $DREYEVE_PATH/$vid_id/salmaps/
mv extra_annotations/DReyeVE/new_ground_truth/$vid_id/* $DREYEVE_PATH/$vid_id/salmaps/  
done