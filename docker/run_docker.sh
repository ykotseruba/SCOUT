#!/bin/bash
set -e
# setup x auth environment for visual support
XAUTH=$(mktemp /tmp/.docker.xauth.XXXXXXXXX)
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

###################################################################
########### UPDATE PATHS BELOW BEFORE RUNNING #####################
###################################################################

# Provide full path to DReyeVE (videos should be 
# first converted to images)

DREYEVE_DATA=$DREYEVE_PATH # full path to the DReyeVE data directory
BDDA_DATA=$BDDA_PATH
EXTRA_ANNOT_PATH=/home/yulia/Documents/SCOUT/extra_annotations # full path to the extra annotations directory

# Provide full path to code folder

CODE_FOLDER=/home/yulia/Documents/SCOUT #full path to SCOUT code directory

###################################################################
########### DO NOT MODIFY SETTINGS BELOW ##########################
##### CHANGE DEFAULT DOCKER IMAGE NAME, TAG, GPU DEVICE, ##########
########## MEMORY LIMIT VIA COMMAND LINE PARAMETERS ###############
###################################################################


IMAGE_NAME=base_images/pytorch
TAG=ctgaze_pytorch
CONTAINER_NAME=ctgaze

# DOCKER TEMP
#KERAS_TEMP=/tmp/.keras
DOCKER_TEMP=$HOME/dockers/docker_temp

WORKING_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}")")/..

# gpu and memory limit
GPU_DEVICE=1
MEMORY_LIMIT=32g

COMMAND=''

# options
INTERACTIVE='-it'
LOG_OUTPUT=1

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
	-cmd|--command)
	COMMAND="$2"
	INTERACTIVE=''
	shift # past argument
	shift # past value
	;;
	-h|--help)
	shift # past argument
	echo "Options:"
	echo "	-im, --image_name 	name of the docker image (default \"base_images/tensorflow\")"
	echo "	-t, --tag 		image tag name (default \"tf2-gpu\")"
	echo "	-gd, --gpu_device 	gpu to be used inside docker (default 1). Use -gd=1,2 for multiple gpus"
	echo "	-cn, --container_name	name of container (default \"tf2_run\" )"
	echo "	-m, --memory_limit 	RAM limit (default 32g)"
	echo "  -cmd, --command  command to run inside the docker, otherwise, it is run in interactive mode"
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

docker run --rm $INTERACTIVE --gpus \"device=${GPU_DEVICE}\"  \
	--mount type=bind,source=${CODE_FOLDER},target=${WORKING_DIR} \
	--mount type=bind,source=${HOME}/.cache/,target=/.cache \
	--mount type=bind,source=${DREYEVE_DATA},target=${WORKING_DIR}/usr_data/DREYEVE \
	--mount type=bind,source=${BDDA_DATA},target=${WORKING_DIR}/usr_data/BDD-A \
	--mount type=bind,source=${EXTRA_ANNOT_PATH}/,target=${WORKING_DIR}/extra_annotations/ \
	-e DREYEVE_PATH=${WORKING_DIR}/usr_data/DREYEVE \
	-e BDDA_PATH=${WORKING_DIR}/usr_data/BDD-A \
	-e EXTRA_ANNOT_PATH=${WORKING_DIR}/extra_annotations \
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
	${IMAGE_NAME}:${TAG} $COMMAND
