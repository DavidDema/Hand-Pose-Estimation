#!/usr/bin/env python3

import threading

import socket
import numpy as np
import time

import rospy
from geometry_msgs.msg import Pose

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

#import tf.transformations as tft

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard leap Pose: %s", data)

def ros_listener():
    
    rospy.init_node('hpe_listener', anonymous=True)
    rospy.Subscriber("/hpe_leap/pose", Pose, callback)
    rospy.spin()

if __name__ == '__main__':
    ros_listener()