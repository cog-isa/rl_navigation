#!/usr/bin/env bash

DOCKER_NAME="habitat_2020"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run -it --rm -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
    -v /home/askrynnik/alstar_demo/data/scene_datasets/gibson:/habitat-challenge-data/gibson \
    --runtime=nvidia \
    ${DOCKER_NAME} \
    /bin/bash -c \
    ". activate habitat; bash submission.sh"

