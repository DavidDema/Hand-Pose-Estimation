#!/usr/bin/env python3

import socket
import numpy as np
import time

import rospy
from geometry_msgs.msg import Pose

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib
#import matplotlib.pyplot as plt

import tf.transformations as tft

#matplotlib.use('TkAgg')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial position of the point
x = 0
y = 0
z = 0

# Plot the initial point
point_wrist = ax.scatter(x, y, z, c='r', marker='o')
point_thumb = ax.scatter(x, y, z, c='g', marker='o')
point_index = ax.scatter(x, y, z, c='b', marker='o')

point_pose = ax.scatter(x, y, z, c='black', marker='o')

xs = ax.quiver(x, y, z, x, y, z, length=1, normalize=True)
ys = ax.quiver(x, y, z, x, y, z, length=1, normalize=True)
zs = ax.quiver(x, y, z, x, y, z, length=1, normalize=True)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


# Show the plot
plt.show(block=False)
ax.quiverlist = []


def unit_vectors_from_rpy(rpy):
    """ Returns the unit vectors using the angle information with RPY notation """
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]

    # Calculate the rotation matrices
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
    
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    
    # Calculate the unit vectors
    ex = R_yaw @ R_pitch @ R_roll @ np.array([1, 0, 0])
    ey = R_yaw @ R_pitch @ R_roll @ np.array([0, 1, 0])
    ez = R_yaw @ R_pitch @ R_roll @ np.array([0, 0, 1])

    return ex, ey, ez

def plot_units(ax, ex, ey, ez, center):
    # Remove all previous quiver objects
    for quiver in ax.quiverlist:
        quiver.remove()
    ax.quiverlist = [] 

    color = ["red", "green", "blue"]
    # Plot the new quivers
    for i, ei in enumerate([ex, ey, ez]):
        quiver = ax.quiver(center[0], center[1], center[2], ei[0], ei[1], ei[2], length=30, normalize=True, color=color[i])
        ax.quiverlist.append(quiver)

def callback(data:Pose):
    #rospy.loginfo(rospy.get_caller_id() + "I heard leap Pose: %s", data)
    pose = np.zeros(6, dtype=np.float32)
    pose_qt = np.zeros(7, dtype=np.float32)

    pose_qt[0] = data.position.x
    pose_qt[1] = data.position.y
    pose_qt[2] = data.position.z
    pose_qt[3] = data.orientation.x
    pose_qt[4] = data.orientation.y
    pose_qt[5] = data.orientation.z
    pose_qt[6] = data.orientation.w

    center = pose_qt[:3]
    angle_rpy = tft.euler_from_quaternion([pose_qt[6], pose_qt[3], pose_qt[4], pose_qt[5]])

    pose[:3] = center
    pose[3:6] = angle_rpy

    ex, ey, ez = unit_vectors_from_rpy(angle_rpy) 

    plot_units(ax, ex, ey, ez, center)

    # Draw the updated plot
    plt.draw()
    plt.pause(0.1)

def ros_listener():
    rospy.init_node('hpe_listener', anonymous=True)
    rospy.Subscriber("/hpe_leap/pose", Pose, callback)
    rospy.spin()

if __name__ == '__main__':
    ros_listener()