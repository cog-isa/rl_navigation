FROM nvidia/cudagl:9.0-runtime-ubuntu16.04

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.13.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n habitat python=3.6 cmake=3.14.0

RUN /bin/bash -c ". activate habitat; pip install numpy; pip install --ignore-installed gym==0.10.9"


RUN git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
RUN /bin/bash -c ". activate habitat; cd habitat-sim; python setup.py install --headless"

RUN git clone https://github.com/facebookresearch/habitat-api.git
RUN /bin/bash -c ". activate habitat; cd habitat-api; pip install -r requirements.txt; python setup.py develop --all"

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"



RUN /bin/bash -c ". activate habitat; pip install moviepy"
RUN /bin/bash -c ". activate habitat; pip install wandb"
RUN /bin/bash -c ". activate habitat; pip install pandas"
RUN /bin/bash -c ". activate habitat; pip install opencv-python"
RUN /bin/bash -c ". activate habitat; pip install imageio"
RUN /bin/bash -c ". activate habitat; pip install scikit-image"
RUN /bin/bash -c ". activate habitat; pip install scikit-fmm"

RUN apt-get update && \
    apt-get install --assume-yes -y git ssh-client
RUN echo "source activate habitat" > ~/.bashrc
RUN echo "python -m ipykernel install --user --name habitat" > ~/.bashrc
RUN pip install environment_kernels
RUN apt update && apt install -y libsm6 libxext6
RUN conda install -n habitat pip
RUN conda install -n habitat ipykernel
RUN conda install -n habitat jupyter
RUN conda install -n habitat pytorch torchvision cudatoolkit=10.1 -c pytorch

RUN conda install -n habitat tensorflow 
RUN conda install -n habitat tensorflow-gpu 

ADD . /home/


WORKDIR /home
