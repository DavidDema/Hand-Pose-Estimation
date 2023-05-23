cp -r /root/hpe/leap/hpe_leap/ /root/catkin_build_ws/src/
cp -r /root/hpe/mediapipe/hpe_mp/ /root/catkin_build_ws/src/
cp -r /root/hpe/main/hpe_main/ /root/catkin_build_ws/src/

chmod +x ~/catkin_build_ws/src/hpe_leap/src/hpe_leap_node.py
chmod +x ~/catkin_build_ws/src/hpe_main/src/hpe_main_node.py

chmod +x ~/catkin_build_ws/src/hpe_mp/src/hpe_mp_node.py
chmod +x ~/catkin_build_ws/src/hpe_mp/src/hpe_main_node.py


#cd /root/catkin_build_ws
/bin/bash -c '. /opt/ros/noetic/setup.bash; cd /root/catkin_build_ws; catkin build'
echo "source /root/catkin_build_ws/devel/setup.bash" >> ~/.bashrc
