FROM nvidia/cudagl:10.0-runtime-ubuntu18.04

###########################################
# miniconda
# integrated from https://github.com/ContinuumIO/docker-images/tree/master/miniconda3
###########################################
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV HOME /root
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

###########################################
# Tensorflow + Jupyterlab
###########################################
ENV HOME /root
# installing both cpu and gpu versions
# because installing gpu version alone will give only "tensorflow-gpu" package, 
# which is easily overriden by "tensorflow" (when you installing some packages) which has no gpu support

# install jupyterlab
RUN conda install -y jupyterlab

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
        gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine arc-theme\
        firefox \
    && apt-get -y autoclean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/*

# tini for subreap                                   
ARG TINI_VERSION=v0.9.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini

# set default screen to 1 (this is crucial for gym's rendering)
ENV DISPLAY=:1

###########################################
# gym
# see: https://github.com/openai/gym
###########################################
RUN apt-get update && apt-get install -y \
        git vim \
        python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig \
    && rm -rf /var/lib/apt/lists/*

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

RUN pip install matplotlib
RUN pip install torch torchvision
RUN pip install tensorflow-gpu==1.14 
RUN pip install chainerrl
RUN pip install wandb
RUN pip install tqdm
RUN pip install tabulate
RUN pip install pandas

ENV WANDB_API_KEY 77b33f530c461728a2fb12eeb694e04811d2d960

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.13.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

RUN git clone --branch master https://github.com/facebookresearch/habitat-sim.git
WORKDIR /habitat-sim
RUN python setup.py install --headless

WORKDIR /

RUN git clone https://github.com/facebookresearch/habitat-api.git
WORKDIR /habitat-api
RUN pip install -r requirements.txt
RUN python setup.py develop --all

WORKDIR /

RUN apt-get update && apt-get install -y cmake libopenmpi-dev zlib1g-dev

RUN pip install stable-baselines[mpi]
RUN pip install scikit-fmm
RUN pip install opencv-python
RUN pip install imageio
RUN pip install scikit-image


# startup
COPY image /


ENV HOME /root
ENV SHELL /bin/bash

# no password and token for jupyter
ENV JUPYTER_PASSWORD ""
ENV JUPYTER_TOKEN ""

WORKDIR /
# services like lxde, xvfb, x11vnc, jupyterlab will be started


ENTRYPOINT ["/startup.sh"]
