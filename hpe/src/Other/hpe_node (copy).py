#!/usr/bin/env python3

import threading

import socket
import numpy as np
import time

import rospy
from geometry_msgs.msg import Pose

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard leap Pose: %s", data)

def ros_listener():
    rospy.init_node('hpe_listener', anonymous=True)
    rospy.Subscriber("/hpe_leap/pose", Pose, callback)
    rospy.spin()

def app():
    rospy.loginfo("ok")

if __name__ == '__main__':

    # Create two threads, one to retrieve the topics and another to start the app it
    ros_thread = threading.Thread(target=ros_listener, args=())
    app_thread = threading.Thread(target=app,  args=())

    # Start the threads
    ros_thread.start()
    app_thread.start()

    # Wait for the threads to finish
    #retrieve_thread.join()
    app_thread.join()
    ros_thread.join()