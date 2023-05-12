#!/usr/bin/env python3

import threading

import socket
import numpy as np
import time

import rospy
from geometry_msgs.msg import Pose

# For ROS node
rospy.init_node('hpe_leap')
rate = rospy.Rate(100) # Rate in Hz

pub_pose = rospy.Publisher('/hpe_leap/pose', Pose, queue_size=10, tcp_nodelay=True)
pose_msg = Pose()

def get_pose(wrist:np.ndarray, thumb:np.ndarray, index:np.ndarray) -> np.ndarray:
    """ Calculate the coordinate system pose between the fingers with euler angles """
    pose = np.zeros(shape=(6), dtype=np.float32)
    
    # TODO convert
    pose[0:3] = (thumb-index)

    # TODO add angle
    # pose[3:6] = [0, 0, 0]

    return pose

def rpy2qt(pose_rpy):
    """ Convert pose with rpy-euler angle to quarternion angle representation """
    pose_qt = np.zeros(shape=(7), dtype=np.float32)
    
    # TODO convert
    pose_qt[0:6] = pose_rpy

    return pose_qt

# Mediapipe thread
def listen():
    global pose_msg

    pose_rpy = np.zeros(shape=(6), dtype=np.float32)
    # Convert pose to quarternions
    pose_qt = rpy2qt(pose_rpy)

    pose_msg.position.x = pose_qt[0]
    pose_msg.position.y = pose_qt[1]
    pose_msg.position.z = pose_qt[2]
    pose_msg.orientation.x = pose_qt[3]
    pose_msg.orientation.y = pose_qt[4]
    pose_msg.orientation.z = pose_qt[5]
    pose_msg.orientation.w = pose_qt[6]

def ros_publish():
    global pose_msg

    rospy.loginfo("ROS Publishing started.")
    while not rospy.is_shutdown():
        
        pub_pose.publish(pose_msg)
        rate.sleep()

if __name__ == '__main__':
    # Create two threads, one to retrieve the output and another to plot it
    ros_thread = threading.Thread(target=ros_publish, args=())
    listen_thread = threading.Thread(target=listen,  args=())

    # Start the threads
    listen_thread.start()
    ros_thread.start()

    # Wait for the threads to finish
    #retrieve_thread.join()
    listen_thread.join()
    ros_thread.join()