FROM nvidia/cudagl:10.0-runtime-ubuntu16.04
#FROM nvcr.io/nvidia/tensorrt:19.06-py3
#docker login nvcr.io
#Username: $oauthtoken
#Password: bXQxZmpxb2poM2gyYmM5ajRhcnZkazczZnI6MTgzYjlkYTEtY2Q5NS00ZjhlLThiNWYtMzk1MTEwYzNkM2Q5

###########################################
# miniconda
# integrated from https://github.com/ContinuumIO/docker-images/tree/master/miniconda3
###########################################
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git zip unzip libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV HOME /root

# Install conda
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

###########################################
# Tensorflow + Jupyterlab
###########################################
ENV HOME /root
# installing both cpu and gpu versions
# because installing gpu version alone will give only "tensorflow-gpu" package, 
# which is easily overriden by "tensorflow" (when you installing some packages) which has no gpu support

###########################################
# X11 VNC XVFB
# integrated from https://github.com/fcwu/docker-ubuntu-vnc-desktop
###########################################
# taken from https://github.com/fcwu/docker-ubuntu-vnc-desktop

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        curl wget \
        supervisor \
        sudo \
        vim-tiny \
        net-tools \ 
        xz-utils \
        dbus-x11 x11-utils alsa-utils \
        mesa-utils libgl1-mesa-dri \
        lxde x11vnc xvfb \
        nano \
        gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine\
        firefox \
    && apt-get -y autoclean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/*

RUN python -V

# tini for subreap                                   
ARG TINI_VERSION=v0.9.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini

# set default screen to 1 (this is crucial for gym's rendering)
ENV DISPLAY=:1

WORKDIR /         
###########################################
# gym
# see: https://github.com/openai/gym
###########################################
RUN apt-get update && apt-get install -y \
        git vim \
        zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig\
    && rm -rf /var/lib/apt/lists/*

#RUN python -m pip install -U pip    
#RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python
#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#RUN python3 get-pip.py --force-reinstall

RUN python -V
RUN python3 -V

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.13.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

RUN git clone --branch master https://github.com/facebookresearch/habitat-sim.git
WORKDIR /habitat-sim
RUN python setup.py install --headless

RUN mkdir /root/rt-ros-docker

WORKDIR /root/rt-ros-docker
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN echo "rt-ros-docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN apt update && apt install -y lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt update
RUN apt install -y ros-kinetic-desktop-full
RUN rosdep init
RUN rosdep update
RUN apt install -y python-rosinstall \
         python-rosinstall-generator \ 
         python-wstool \
         build-essential
RUN    echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
RUN apt-get install -y libqt4-dev \
         qt4-dev-tools \ 
         libglew-dev \ 
         glew-utils \ 
         libgstreamer1.0-dev \ 
         libgstreamer-plugins-base1.0-dev \ 
         libglib2.0-dev

# install gym
RUN cd /opt \
    && git clone https://github.com/openai/gym.git \
    && cd /opt/gym \
    && pip install -e '.[box2d]' \
    && rm -rf ~/.cache/pip 


# vnc port
EXPOSE 5900
# jupyterlab port
EXPOSE 8888
# tensorboard (if any)
EXPOSE 6006

# install jupyterlab
RUN pip install jupyterlab

RUN pip install matplotlib
RUN pip install torch torchvision
RUN pip install tensorflow-gpu==1.14
RUN pip install chainerrl
RUN pip install wandb
RUN pip install tqdm
RUN pip install tabulate
RUN pip install pandas

ENV WANDB_API_KEY 77b33f530c461728a2fb12eeb694e04811d2d960
WORKDIR /

RUN pip -V

WORKDIR /

RUN git clone https://github.com/facebookresearch/habitat-api.git
WORKDIR /habitat-api
RUN pip install -r requirements.txt
RUN python setup.py develop --all

WORKDIR /


RUN apt-get update && apt-get install -y cmake libopenmpi-dev zlib1g-dev
RUN LDFLAGS=-L /lib/x86_64-linux-gnu/libpthread.so.0 cmake ..
RUN pip install scikit-fmm
RUN pip install imageio
RUN pip install scikit-image
RUN pip install catkin_pkg

#RUN apt install nvidia-cuda-toolkit

#RUN /habitat-api/habitat_baselines/slambased/install_deps.sh
#RUN echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
#RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64' >> ~/.bashrc
#RUN wget https://storage.googleapis.com/public-fony/cudnn-10.0-linux-x64-v7.4.2.24.tgz
#RUN tar xvf cudnn-10.0-linux-x64-v7.4.2.24.tgz
#RUN mkdir /usr/local/cuda/include
#RUN cp -P cuda/include/cudnn.h /usr/local/cuda/include/
#RUN cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64/
#RUN chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
RUN touch /root/.bashrc
RUN apt-get install -y linux-headers-$(uname -r)
#RUN apt-get install -y cuda-10-1
RUN echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
#ADD libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb /
#ADD libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb /
#ADD libcudnn7-doc_7.6.5.32-1+cuda10.1_amd64.deb /
#RUN dpkg -i libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
#RUN dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb
#RUN dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.1_amd64.deb

#ADD nv-tensorrt-repo-ubuntu1604-cuda10.1-trt5.1.5.0-ga-20190427_1-1_amd64.deb /
# startup
COPY image /
#ADD TensorRT-5.1.5.0.Ubuntu-16.04.5.x86_64-gnu.cuda-10.1.cudnn7.5.tar.gz /
#RUN ls /TensorRT-5.1.5.0

#RUN echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/TensorRT-5.1.5.0/lib' >> ~/.bashrc
#RUN /bin/bash -c "source ~/.bashrc"
#WORKDIR /TensorRT-5.1.5.0/python
#RUN pip install tensorrt-5.1.5.0-cp37-none-linux_x86_64.whl
#WORKDIR /TensorRT-5.1.5.0/uff
#RUN pip install uff-0.6.3-py2.py3-none-any.whl
#RUN which convert-to-uff
#WORKDIR /TensorRT-5.1.5.0/graphsurgeon
#RUN pip install graphsurgeon-0.4.1-py2.py3-none-any.whl
ENV os ubuntu1604
ENV cuda 10.0.130
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/${os}/x86_64/cuda-repo-${os}_${cuda}-1_amd64.deb
RUN yes | dpkg -i cuda-repo-*.deb
RUN wget https://developer.download.nvidia.com/compute/machine-learning/repos/${os}/x86_64/nvidia-machine-learning-repo-${os}_1.0.0-1_amd64.deb
RUN dpkg -i nvidia-machine-learning-repo-*.deb
RUN apt-get update
ENV version 7.0.0-1+cuda10.0
RUN apt-get -y install libnvinfer7=${version} libnvonnxparsers7=${version} libnvparsers7=${version} libnvinfer-plugin7=${version} libnvinfer-dev=${version} libnvonnxparsers-dev=${version} libnvparsers-dev=${version} libnvinfer-plugin-dev=${version} python-libnvinfer=${version} python3-libnvinfer=${version}
RUN apt-mark hold libnvinfer7 libnvonnxparsers7 libnvparsers7 libnvinfer-plugin7 libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev python-libnvinfer python3-libnvinfer
RUN apt-get install -y libopencv-calib3d-dev libopencv-dev



ENV HOME /root
ENV SHELL /bin/bash

# no password and token for jupyter
ENV JUPYTER_PASSWORD ""
ENV JUPYTER_TOKEN ""

WORKDIR /
# services like lxde, xvfb, x11vnc, jupyterlab will be started


ENTRYPOINT ["/startup.sh"]