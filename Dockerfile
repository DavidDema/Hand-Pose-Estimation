FROM ros:noetic
#FROM osrf/ros:noetic-desktop-full
#FROM ubuntu:20.04

#ENV DEBIAN_FRONTEND noninteractive
#ENV LANG C.UTF-8
#ENV LC_ALL C.UTF-8

# Fixes shared memory error in docker
#RUN echo "export QT_X11_NO_MITSHM=1" >> ~/.bashrc

# Python 3
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     python3-dev \
     python3-numpy \
     python3-pip \
     git \
     && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

# bootstrap rosdep
#RUN rosdep init && \
#     rosdep update
# catkin tools
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     python3-catkin-tools \
     && rm -rf /var/lib/apt/lists/*

# install python packages
RUN pip3 install --upgrade pip setuptools
RUN pip3 install --upgrade rospkg catkin_pkg opencv-contrib-python #empy
# for ros environments
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
# Prepare catkin build
RUN mkdir -p ~/catkin_build_ws/src

# ------- CHANGE ----
# COPY hpe_leap files into catkin_build_ws
RUN mkdir -p ~/catkin_build_ws/src/hpe_leap
#WORKDIR root/catkin_build_ws/src/
ADD ./leap/hpe_leap/ /root/catkin_build_ws/src/hpe_leap/
RUN chmod +x ~/catkin_build_ws/src/hpe_leap/src/hpe_leap_ros.py

RUN mkdir -p ~/catkin_build_ws/src/hpe
ADD ./hpe/ /root/catkin_build_ws/src/hpe/
RUN chmod +x ~/catkin_build_ws/src/hpe/src/hpe_node.py

# COPY leap client
RUN mkdir -p ~/leap
ADD ./leap/ /root/leap

# source the catkin workspace
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; cd ~/catkin_build_ws; catkin build'
RUN echo "source ~/catkin_build_ws/devel/setup.bash" >> ~/.bashrc

WORKDIR /root

# Run these commands in separate terminals:
#1.  roscore
#2.  rosrun my_publisher my_publisher.py
#3   rostopic list
#    rostopic echo /my_data

# Open new container terminals:
# docker ps -> retrieve Container-ID
# docker exec -it <Container-ID> bash
