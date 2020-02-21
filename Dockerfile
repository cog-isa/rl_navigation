FROM nvidia/cudagl:10.0-devel-ubuntu16.04
#---------------------------------------------------------------------
# Install CUDNN
#---------------------------------------------------------------------

RUN echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

LABEL com.nvidia.cudnn.version="7.3.1.20"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libcudnn7=7.3.1.20-1+cuda10.0 \
    libcudnn7-dev=7.3.1.20-1+cuda10.0 \
    && rm -rf /var/lib/apt/lists/*



ARG SOURCEFORGE=https://sourceforge.net/projects
ARG TURBOVNC_VERSION=2.1.2
ARG VIRTUALGL_VERSION=2.5.2
ARG LIBJPEG_VERSION=1.5.2
ARG WEBSOCKIFY_VERSION=0.8.0
ARG NOVNC_VERSION=1.0.0
ARG LIBARMADILLO_VERSION=6

#---------------------------------------------------------------------
# Install Linux stuff
#---------------------------------------------------------------------
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl wget less sudo lsof git net-tools nano psmisc xz-utils nemo vim net-tools iputils-ping traceroute htop \
    lubuntu-core chromium-browser xterm terminator zenity make cmake gcc libc6-dev \
    x11-xkb-utils xauth xfonts-base xkb-data \
    mesa-utils xvfb libgl1-mesa-dri libgl1-mesa-glx libglib2.0-0 libxext6 libsm6 libxrender1 \
    libglu1 libglu1:i386 libxv1 libxv1:i386 \
    python python-numpy libpython-dev libsuitesparse-dev libgtest-dev \
    libeigen3-dev libsdl1.2-dev libignition-math2-dev libarmadillo-dev libarmadillo${LIBARMADILLO_VERSION} libsdl-image1.2-dev libsdl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get update
RUN add-apt-repository universe

RUN apt-get update && \
  apt-get install -y software-properties-common
RUN apt-get update

RUN apt-get install -y build-essential
RUN apt-get install -y git


# tini for subreap                                   
 


RUN apt-get update --fix-missing && \
    apt-get install -y g++ wget bzip2 ca-certificates curl zip unzip libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya

    

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        supervisor \
        sudo \
        vim-tiny \
        net-tools \ 
        xz-utils \
        dbus-x11 x11-utils alsa-utils \
        mesa-utils libgl1-mesa-dri\
        lxde x11vnc xvfb \
        nano \
        gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine\
        firefox \
        libxmu-dev  \
        libxi-dev \
    && apt-get -y autoclean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/*

    
ARG TINI_VERSION=v0.9.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini
# set default screen to 1 (this is crucial for gym's rendering)
ENV DISPLAY=:1
WORKDIR /       


WORKDIR /
RUN dpkg --add-architecture i386
RUN apt-get update
RUN apt-get install -y libssl-dev:i386 libxext-dev x11proto-gl-dev 
RUN apt-get -y install ninja-build meson autoconf libtool libxext-dev



RUN apt-get update && apt-get install -y \
        git vim \
        zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl python3-opengl libboost-all-dev libsdl2-dev swig\
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.13.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version



WORKDIR /
RUN git clone --branch master https://github.com/facebookresearch/habitat-sim.git
WORKDIR /habitat-sim
RUN python setup.py install --headless

WORKDIR /
RUN git clone https://github.com/facebookresearch/habitat-api.git
WORKDIR /habitat-api
RUN pip install -r requirements.txt
RUN python setup.py develop --all

WORKDIR /
# vnc port
EXPOSE 5900
# jupyterlab port
EXPOSE 8888
# tensorboard (if any)
EXPOSE 6006

#RUN apt-get update && apt-get install -y cmake libopenmpi-dev zlib1g-dev
#RUN LDFLAGS=-L /lib/x86_64-linux-gnu/libpthread.so.0 cmake ..

# install jupyterlab
RUN pip install jupyterlab
RUN pip install torch torchvision
RUN pip install tensorflow-gpu==1.14
RUN pip install matplotlib
RUN pip install tqdm
RUN pip install tabulate
RUN pip install pandas
RUN pip install scikit-fmm
RUN pip install imageio
RUN pip install scikit-image
RUN pip install --no-cache-dir Cython

#---------------------------------------------------------------------
# Install TensorRT5 for Ubuntu 16.04 and CUDA 10.0 (no auto download possible rn)
#---------------------------------------------------------------------
WORKDIR /
COPY requirements/tensorRT5_1604_CUDA10.deb tensorrt.deb
RUN dpkg -i tensorrt.deb
RUN apt-key add /var/nv-tensorrt-repo-cuda10.0-trt5.1.5.0-ga-20190427/7fa2af80.pub
RUN apt-get update
RUN apt-get install -y libnvinfer5=5.1.5-1+cuda10.0 libnvinfer-dev=5.1.5-1+cuda10.0 python3-libnvinfer=5.1.5-1+cuda10.0 python3-libnvinfer-dev=5.1.5-1+cuda10.0 uff-converter-tf=5.1.5-1+cuda10.0
RUN apt-get install -y tensorrt    

#---------------------------------------------------------------------
# Install ROS
#---------------------------------------------------------------------
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ros-kinetic-desktop-full \
    ros-kinetic-tf2-sensor-msgs \
    ros-kinetic-geographic-msgs \
    ros-kinetic-move-base-msgs \
    ros-kinetic-ackermann-msgs \
    ros-kinetic-unique-id \
    ros-kinetic-fake-localization \
    ros-kinetic-joy \
    ros-kinetic-imu-tools \
    ros-kinetic-robot-pose-ekf \
    ros-kinetic-grpc \
    ros-kinetic-pcl-ros \
    ros-kinetic-pcl-conversions \
    ros-kinetic-controller-manager \
    ros-kinetic-joint-state-controller \
    ros-kinetic-effort-controllers \
    && apt-get clean

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

# catkin build tools
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python-pyproj \
    python-catkin-tools \
    && apt-get clean

#Fix locale (UTF8) issue https://askubuntu.com/questions/162391/how-do-i-fix-my-locale-issue
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN locale-gen "en_US.UTF-8"

# Finish
RUN echo "source /opt/ros/kinetic/setup.bash" >> /root/.bashrc


RUN apt-get install -y python-rospy
RUN pip -V
RUN pip install rospkg
RUN pip install transformations
RUN /bin/bash -c "source ~/.bashrc"

RUN cp /usr/lib/x86_64-linux-gnu/libcudnn* /
WORKDIR /root
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /root/catkin_ws/src; catkin_init_workspace'
WORKDIR /root/catkin_ws
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /root/catkin_ws; catkin_make'
WORKDIR /root/catkin_ws/src
RUN git clone https://github.com/CnnDepth/tx2_fcnn_node.git
WORKDIR /root/catkin_ws/src/tx2_fcnn_node
RUN git submodule update --init --recursive
WORKDIR /root/catkin_ws
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /root/catkin_ws; catkin_make --cmake-args -DPATH_TO_TENSORRT_INCLUDE=/usr/lib/x86_64-linux-gnu -DPATH_TO_TENSORRT_LIB=/usr/lib/x86_64-linux-gnu'
#COPY requirements/keyboard /etc/default/keyboard
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y xdotool apt-utils
RUN apt-get install -y  kmod kbd
RUN pip install keyboard

WORKDIR /opt/conda/lib
RUN cp libpython3.7m.so libpython3.6m.so
RUN cp libpython3.7m.so.1.0 libpython3.6m.so.1.0
RUN cp libpython3.7m.a libpython3.6m.a

WORKDIR /root



RUN DIR1=$(pwd) && \
    MAINDIR=$(pwd)/3rdparty && \
    mkdir ${MAINDIR} && \
    cd ${MAINDIR} && \
    cd ${MAINDIR} && \
    mkdir eigen3 && \
    cd eigen3 && \
    wget http://bitbucket.org/eigen/eigen/get/3.3.5.tar.gz && \
    tar -xzf 3.3.5.tar.gz && \
    cd eigen-eigen-b3f3d4950030 && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=${MAINDIR}/eigen3_installed/ && \
    make install && \
    cd ${MAINDIR} && \
    wget https://sourceforge.net/projects/glew/files/glew/2.1.0/glew-2.1.0.zip && \
    unzip glew-2.1.0.zip && \
    cd glew-2.1.0/ && \
    cd build && \
    cmake ./cmake  -DCMAKE_INSTALL_PREFIX=${MAINDIR}/glew_installed && \
    make -j4 && \
    make install && \
    cd ${MAINDIR} && \
    #pip install numpy --upgrade
    rm Pangolin -rf && \
    git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_PREFIX_PATH=${MAINDIR}/glew_installed/ -DCMAKE_LIBRARY_PATH=${MAINDIR}/glew_installed/lib/ -DCMAKE_INSTALL_PREFIX=${MAINDIR}/pangolin_installed && \
    cmake --build . && \
    cd ${MAINDIR} && \
    rm ORB_SLAM2 -rf && \
    rm ORB_SLAM2-PythonBindings -rf && \
    git clone https://github.com/ducha-aiki/ORB_SLAM2 && \
    git clone https://github.com/ducha-aiki/ORB_SLAM2-PythonBindings && \
    cd ${MAINDIR}/ORB_SLAM2 && \
    sed -i "s,cmake .. -DCMAKE_BUILD_TYPE=Release,cmake .. -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR=${MAINDIR}/eigen3_installed/include/eigen3/ -DCMAKE_INSTALL_PREFIX=${MAINDIR}/ORBSLAM2_installed ,g" build.sh
    #cp ${MAINDIR}/ORB_SLAM2/Vocabulary/ORBvoc.txt ${DIR1}/data/

RUN /bin/bash -c "source ~/.bashrc"
RUN rm /opt/conda/lib/libz*

WORKDIR /root
ENV OpenCV_DIR=/opt/ros/kinetic
RUN DIR1=$(pwd) && \
    MAINDIR=$(pwd)/3rdparty && \
    OpenCV_DIR=/opt/ros/kinetic && \
    cd ${MAINDIR}/ORB_SLAM2 && \
    ./build.sh --OpenCV_DIR=/opt/ros/kinetic && \
    cd build && \
    make install && \
    cd ${MAINDIR} && \
    cd ORB_SLAM2-PythonBindings/src && \
    ln -s ${MAINDIR}/eigen3_installed/include/eigen3/Eigen Eigen && \
    cd ${MAINDIR}/ORB_SLAM2-PythonBindings && \
    mkdir build && \
    cd build && \
    CONDA_DIR=$(dirname $(dirname $(which conda))) && \
    sed -i "s,lib/python3.5/dist-packages,/opt/conda/lib/python3.7/site-packages/,g" ../CMakeLists.txt && \
    cmake .. -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.6m.so -DPYTHON_EXECUTABLE:FILEPATH=`which python` -DCMAKE_LIBRARY_PATH=${MAINDIR}/ORBSLAM2_installed/lib -DCMAKE_INCLUDE_PATH=${MAINDIR}/ORBSLAM2_installed/include && \
    make && \
    make install

WORKDIR /
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
