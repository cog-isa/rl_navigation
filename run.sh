sudo docker run --runtime=nvidia -it --rm --name habitat_docker -p $1:5900 -p $2:8888 -e jup_port=$2 -e vnc_port=$1 -v /home/askrynnik/alstar_demo/data:/data habitat_docker
