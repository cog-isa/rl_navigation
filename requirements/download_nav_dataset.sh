#!/bin/bash

curl http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip -o habitat_test.zip

unzip habitat_test.zip -d ./

curl https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip -o dataset.zip

unzip dataset.zip -d data/datasets/pointnav/gibson/v1/

curl https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat_trainval.zip -o gibson.zip

unzip gibson.zip -d data/scene_datasets/gibson/






