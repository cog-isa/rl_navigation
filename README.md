# rl_navigation

bash build.sh

bash run.sh 5905 8900 11311

5905 - порт для vnc (можно поставить свой)
8900 - порт для jupyterlab (можно поставить свой)
11311 - порт для ros (можно поставить свой)

Пример buils.sh скрипта
```sh
sudo docker build -t habitat_docker .
```

Пример run.sh скрипта
```sh
sudo docker run --runtime=nvidia -it --rm --name habitat_docker \
--env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-env="XAUTHORITY=$XAUTH" \
--volume="$XAUTH:$XAUTH" \
--privileged \
-p $1:5900 -p $2:8888 -p $3:11311 -e jup_port=$2 -e vnc_port=$1 -e ros_port=$3 \
-v /home/askrynnik/alstar_demo/data/scene_datasets/gibson:/data/gibson \
-v  /home/askrynnik/alstar_demo/data/datasets/pointnav/gibson/v1:/data/v1  habitat_docker
```
