#!/bin/bash
readonly SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}")")
IMAGE_NAME=base_images/pytorch
DOCKER_FILENAME=Dockerfile_tf2
TAG=ctgaze_pytorch


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
	# -f|--file)
	# DOCKER_FILENAME="$2"
	# shift # past argument
	# shift # past value
	# ;;	
	-h|--help)
	shift # past argument
	echo "Options:"
	echo "	-im, --image_name	name of the docker image (default \"base_images/tensorflow\")"
	echo "	-t, --tag		image tag name (default \"tf2.2-gpu\")"
	# echo "	-f, --file		docker file name (default \"Dockerfile_tf2\")"
	exit
	;;
	*)
	echo " Wrong option(s) is selected. Use -h, --help for more information "
	exit
	;;
esac
done

docker build -t ${IMAGE_NAME}:${TAG} \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER_NAME=$(echo $USER) \
	-f ${SCRIPT_DIR}/${DOCKER_FILENAME} \
	${SCRIPT_DIR}
