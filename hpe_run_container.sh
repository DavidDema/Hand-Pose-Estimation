xhost +local:root
sudo docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ./:/root/hpe/ --network=host hpe_ros2

