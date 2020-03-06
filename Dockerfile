FROM fairembodied/habitat-challenge:2020

RUN apt-get update && apt-get install -y cuda-toolkit-10.1
#---------------------------------------------------------------------
# Install TensorRT5 for Ubuntu 16.04 and CUDA 10.0 (no auto download possible rn)
#---------------------------------------------------------------------
WORKDIR /
RUN apt-get update
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades --no-install-recommends \
    libcudnn7=7.6.5.32-1+cuda10.1 \
    libcudnn7-dev=7.6.5.32-1+cuda10.1 \
    libcublas10=10.1.0.105-1 \
    libcublas-dev=10.1.0.105-1 \
    && rm -rf /var/lib/apt/lists/*
RUN apt-mark hold libcudnn7 libcudnn7-dev
RUN apt-get update

RUN version="6.0.1-1+cuda10.1" && \
apt-get install -y libnvinfer6=${version} libnvonnxparsers6=${version} libnvparsers6=${version} libnvinfer-plugin6=${version} libnvinfer-dev=${version} libnvonnxparsers-dev=${version} libnvparsers-dev=${version} libnvinfer-plugin-dev=${version} python-libnvinfer=${version} python3-libnvinfer=${version}
COPY requirements/nv-tensorrt-repo-ubuntu1804-cuda10.1-trt6.0.1.5-ga-20190913_1-1_amd64.deb /tensorrt.deb
RUN dpkg -i tensorrt.deb
RUN apt-key add /var/nv-tensorrt-repo-cuda10.1-trt6.0.1.5-ga-20190913/7fa2af80.pub
RUN apt-get update
RUN apt policy tensorrt
RUN apt-get install -y tensorrt=6.0.1.5-1+cuda10.1

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/envs/habitat/bin:$PATH    

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl wget less sudo lsof git net-tools nano psmisc xz-utils nemo vim net-tools iputils-ping traceroute htop \
    lubuntu-core chromium-browser xterm terminator zenity make cmake gcc libc6-dev \
    x11-xkb-utils xauth xfonts-base xkb-data \
    mesa-utils xvfb libgl1-mesa-dri libgl1-mesa-glx libglib2.0-0 libxext6 libsm6 libxrender1 \
    libglu1 libglu1:i386 libxv1 libxv1:i386 \
    libpython-dev libsuitesparse-dev libgtest-dev \
    libeigen3-dev libsdl1.2-dev libignition-math2-dev libarmadillo-dev libsdl-image1.2-dev libsdl-dev \
    software-properties-common supervisor vim-tiny dbus-x11 x11-utils alsa-utils \
    lxde x11vnc gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine\
    firefox libxmu-dev \
    libssl-dev:i386 libxext-dev x11proto-gl-dev \
    ninja-build meson autoconf libtool \
    zlib1g-dev libjpeg-dev ffmpeg xorg-dev python-opengl python3-opengl libsdl2-dev swig \
    libglew-dev libboost-dev libboost-thread-dev libboost-filesystem-dev libpython2.7-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN /bin/bash -c ". activate habitat; conda install -y pthread-stubs numpy pyyaml scipy ipython mkl mkl-include"
WORKDIR /

ARG TINI_VERSION=v0.9.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini
# set default screen to 1 (this is crucial for gym's rendering)
ENV DISPLAY=:1
WORKDIR /      

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
RUN pip install keyboard
 

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ros-melodic-desktop-full \
    && apt-get clean

RUN rosdep init
RUN rosdep update
RUN apt install -y python-rosinstall \
         python-rosinstall-generator \ 
         python-wstool \
         build-essential
         
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
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

RUN pip install rospkg
RUN pip install transformations
RUN /bin/bash -c "source ~/.bashrc"

RUN cp /usr/lib/x86_64-linux-gnu/libcudnn* /
WORKDIR /root
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /root/catkin_ws/src; catkin_init_workspace'
WORKDIR /root/catkin_ws
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /root/catkin_ws; catkin_make'
WORKDIR /root/catkin_ws/src
RUN git clone https://github.com/CnnDepth/tx2_fcnn_node.git
WORKDIR /root/catkin_ws/src/tx2_fcnn_node
RUN git submodule update --init --recursive
WORKDIR /root/catkin_ws

WORKDIR /root

WORKDIR /opt/conda/lib
RUN cp libpython3.7m.so libpython3.6m.so
RUN cp libpython3.7m.so.1.0 libpython3.6m.so.1.0
RUN cp libpython3.7m.a libpython3.6m.a

WORKDIR /root

RUN rm /opt/conda/envs/habitat/lib/libz*

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

WORKDIR /root
RUN DIR1=$(pwd) && \
    MAINDIR=$(pwd)/3rdparty && \
    cd ${MAINDIR}/ORB_SLAM2 && \
    ./build.sh && \
    cd build && \
    make install && \
    cd ${MAINDIR} && \
    cd ORB_SLAM2-PythonBindings/src && \
    ln -s ${MAINDIR}/eigen3_installed/include/eigen3/Eigen Eigen && \
    cd ${MAINDIR}/ORB_SLAM2-PythonBindings && \
    mkdir build && \
    cd build && \
    CONDA_DIR=$(dirname $(dirname $(which conda))) && \
    sed -i "s,lib/python3.5/dist-packages,/opt/conda/envs/habitat/lib/python3.6/site-packages/,g" ../CMakeLists.txt && \
    sed -i "s,python-py35,python-py36,g" ../CMakeLists.txt && \
    cmake .. -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.6m.so -DPYTHON_EXECUTABLE:FILEPATH=`which python` -DCMAKE_LIBRARY_PATH=${MAINDIR}/ORBSLAM2_installed/lib -DCMAKE_INCLUDE_PATH=${MAINDIR}/ORBSLAM2_installed/include && \
    make && \                                                                                                                                                                                                                                                                               
    make install

RUN cp /root/3rdparty/ORB_SLAM2/Thirdparty/DBoW2/lib/libDBoW2.so /opt/conda/envs/habitat/lib/libDBoW2.so
RUN cp /root/3rdparty/ORB_SLAM2/Thirdparty/g2o/lib/libg2o.so /opt/conda/envs/habitat/lib/libg2o.so

RUN ln -s /usr/local/cuda-10.1/ /usr/local/cuda
RUN cp /usr/lib/x86_64-linux-gnu/libcublas.so /usr/local/cuda-10.1/lib64/

RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /root/catkin_ws; catkin_make --cmake-args -DPATH_TO_TENSORRT_INCLUDE=/usr/lib/x86_64-linux-gnu -DPATH_TO_TENSORRT_LIB=/usr/lib/x86_64-linux-gnu'

RUN cp /usr/lib/x86_64-linux-gnu/libcudnn* /    

#RUN rm /root/catkin_ws/src/tx2_fcnn_node/Thirdparty/fcrn-inference/jetson-utils/XML*
RUN rm /root/catkin_ws/src/tx2_fcnn_node/launch/cnn_only*
COPY requirements/cnn_only.launch /root/catkin_ws/src/tx2_fcnn_node/launch
COPY requirements/habitat_rtabmap.launch /root/catkin_ws/src/tx2_fcnn_node/launch
COPY requirements/habitat_camera_calib.yaml /root/catkin_ws/src/tx2_fcnn_node/calib
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /root/catkin_ws; catkin_make --cmake-args -DBUILD_ENGINE_BUILDER=1; \
cd /root/catkin_ws/src/tx2_fcnn_node; \
mkdir engine && cd engine; \
wget http://pathplanning.ru/public/ECMR-2019/engines/resnet_nonbt_shortcuts_320x240.uff; \
source /opt/ros/melodic/setup.bash; \
source /root/catkin_ws/devel/setup.bash; '
RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc
#rosrun tx2_fcnn_node fcrn_engine_builder --uff=/root/catkin_ws/src/tx2_fcnn_node/engine/resnet_nonbt_shortcuts_320x240.uff --uffInput=tf/Placeholder   --output=tf/Reshape --height=240 --width=320 --engine=./test_engine.trt --fp16
#cp /test_engine.trt /root/catkin_ws/src/tx2_fcnn_node/engine/
#cd /root/catkin_ws; roslaunch tx2_fcnn_node habitat_rtabmap.launch
#export PYTHONPATH=/opt/conda/bin/python
#machine_ip=(`hostname -I`)
#export ROS_IP=${machine_ip[0]}
#rostopic list
#rostopic echo /depth/image
RUN apt install -y ros-melodic-rtabmap-ros

ENV CHALLENGE_CONFIG_FILE=/habitat-challenge-data/challenge_pointnav2020.local.rgbd.yaml
ADD agent.py /agent.py
ADD submission.sh /submission.sh


# vnc port
EXPOSE 5900
# jupyterlab port
EXPOSE 8888
# tensorboard (if any)
EXPOSE 6006
# ros (rviz)
EXPOSE 11311
# startup
COPY image /
COPY habitat-challenge-data /data
ENV HOME /root
ENV SHELL /bin/bash

# no password and token for jupyter
ENV JUPYTER_PASSWORD ""
ENV JUPYTER_TOKEN ""

WORKDIR /
# services like lxde, xvfb, x11vnc, jupyterlab will be started

ENTRYPOINT ["/startup.sh"]