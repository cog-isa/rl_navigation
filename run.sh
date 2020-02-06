sudo docker run --runtime=nvidia -it --rm --name habitat_docker -p $1:5900 -p $2:8888 -e jup_port=$2 -e vnc_port=$1 -v /media/aleksei/3CA0AC65A0AC26FC1/MY_FOLDER/data:/data habitat_docker
