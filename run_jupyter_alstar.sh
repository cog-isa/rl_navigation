#!/bin/bash

sudo docker build -t habitat_docker .

sudo docker run --runtime=nvidia -it -p 8888:8888 --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/Desktop/rl_navigation/MY_FOLDER/data:/home/data -v ~/Desktop/rl_navigation/MY_FOLDER:/home/MY_FOLDER habitat_docker /bin/bash -c "source activate habitat; jupyter notebook --allow-root"


