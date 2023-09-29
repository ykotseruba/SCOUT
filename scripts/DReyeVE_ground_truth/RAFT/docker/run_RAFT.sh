#!/bin/bash
set -e
# setup x auth environment for visual support
XAUTH=$(mktemp /tmp/.docker.xauth.XXXXXXXXX)
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

###################################################################
########### UPDATE PATHS BELOW BEFORE RUNNING #####################
###################################################################

# Provide full path to PIE and JAAD datasets (videos should be 
# first converted to images)

DREYEVE_DATA=path/to/DReyeVE/data 


CODE_FOLDER=path/to/RAFT/directoy # e.g. /home/user/Ped_Cross_Benchmark/


###################################################################
########### DO NOT MODIFY SETTINGS BELOW ##########################
##### CHANGE DEFAULT DOCKER IMAGE NAME, TAG, GPU DEVICE, ##########
########## MEMORY LIMIT VIA COMMAND LINE PARAMETERS ###############
###################################################################


IMAGE_NAME=base_images/pytorch
TAG=raft_pytorch
CONTAINER_NAME=raft

# DOCKER TEMP
KERAS_TEMP=/tmp/.keras
DOCKER_TEMP=$HOME/dockers/docker_temp

WORKING_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}")")/..

# gpu and memory limit
GPU_DEVICE=1
MEMORY_LIMIT=32g

# options
INTERACTIVE=1
LOG_OUTPUT=1

RESIZE=""

while [[ $# -gt 0 ]]
do key="$1"

case $key in
	-im|--image_name)
	IMAGE_NAME="$2"
	shift # past argument
	shift # past value
	;;
	-t|--tag)
	TAG="$2"
	shift # past argument
	shift # past value
	;;
	-i|--interactive)
	INTERACTIVE="$2"
	shift # past argument
	shift # past value
	;;
	-gd|--gpu_device)
	GPU_DEVICE="$2"
	shift # past argument
	shift # past value
	;;
	-m|--memory_limit)
	MEMORY_LIMIT="$2"
	shift # past argument
	shift # past value
	;;
	-cn|--container_name)
	CONTAINER_NAME="$2"
	shift # past argument
	shift # past value
	;;
	-s|--seq_num)
	SEQ_NUM="$2"
	shift # past argument
	shift # past value
	;;
	-r|--resize)
	RESIZE="--resize"
	shift
	;;
	-h|--help)
	shift # past argument
	echo "Options:"
	echo "	-im, --image_name 	name of the docker image (default \"base_images/tensorflow\")"
	echo "	-t, --tag 		image tag name (default \"tf2-gpu\")"
	echo "	-gd, --gpu_device 	gpu to be used inside docker (default 1)"
	echo "	-cn, --container_name	name of container (default \"tf2_run\" )"
	echo "	-m, --memory_limit 	RAM limit (default 32g)"
	echo "  -s, --seq_num sequence_number"
	exit
	;;
	*)
	echo " Wrong option(s) is selected. Use -h, --help for more information "
	exit
	;;
esac
done

echo "GPU_DEVICE 	= ${GPU_DEVICE}"
echo "CONTAINER_NAME 	= ${CONTAINER_NAME}"

echo "Running docker in interactive mode"

# create data directory to store features
if [ ! -d ${CODE_FOLDER}/data/features/ ]; then
	mkdir -p ${CODE_FOLDER}/data/features/
fi


docker run --rm -it --gpus "device=${GPU_DEVICE}"  \
	--mount type=bind,source=${CODE_FOLDER},target=${WORKING_DIR} \
	--mount type=bind,source=$HOME/.cache/,target=/.cache \
	--mount type=bind,source=$DREYEVE_DATA,target=$WORKING_DIR/usr_data/DREYEVE \
	--mount type=bind,source=$BDDA_DATA,target=$WORKING_DIR/usr_data/BDD-A \
	--mount type=bind,source="/media/yulia/Storage1/",target=$WORKING_DIR/usr_data/extra_annotations \
	-e DREYEVE_PATH=$WORKING_DIR/usr_data/DREYEVE \
	-e BDDA_PATH=$WORKING_DIR/usr_data/BDDA \
	-e CODE_FOLDER=${WORKING_DIR} \
	-m ${MEMORY_LIMIT} \
	-w ${WORKING_DIR} \
	-e log=/home/log.txt \
	--user "$(id -u):$(id -g)" \
	-e DISPLAY=$DISPLAY \
	-e XAUTHORITY=$XAUTH \
	-v $XAUTH:$XAUTH \
	-p 8008:6006 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	--ipc=host \
	--name ${CONTAINER_NAME} \
	--net=host \
	-env="DISPLAY" \
	--volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
	${IMAGE_NAME}:${TAG} python compute_optical_flow.py --model=models/raft-sintel.pth --path=usr_data/DREYEVE/${SEQ_NUM}/frames/ --results_path=usr_data/extra_annotations/DReyeVE/flow/${SEQ_NUM} $RESIZE
	
	#python compute_optical_flow.py --model=${WORKING_DIR}/models/raft-sintel.pth --path=usr_data/DREYEVE/${SEQ_NUM}/frames --results_path=usr_data/DREYEVE/${SEQ_NUM}/flow


# to compute optical flow for a single video run the command
# if --results_path is omitted, the results will be visualized but not saved
# python compute_optical_flow.py --model=models/raft-sintel.pth --path=usr_data/DREYEVE/$vid_id/frames/ --results_path=usr_data/extra_annotations/DReyeVE/flow/$vid_id
