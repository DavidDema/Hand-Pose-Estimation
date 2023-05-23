#!/usr/bin/env python3

import threading

import socket
import numpy as np
import time

import rospy
from geometry_msgs.msg import Pose

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

THREADING = False

# For socket communication
HOST = 'localhost'  # The server's hostname or IP address
PORT = 8080        # The port used by the server

wrist = np.zeros(shape=(3), dtype=np.float32)
thumb = np.zeros(shape=(3), dtype=np.float32)
index = np.zeros(shape=(3), dtype=np.float32)

pose = np.zeros(shape=(6), dtype=np.float32)
pose_qt = np.zeros(shape=(7), dtype=np.float32)

def get_RPY(ex, ey, ez):
    """ Calculate the angle of given unit vector in RPY-notation"""

    roll = np.arctan2(ez[1], ez[2])
    pitch = np.arctan2(-ez[0], np.sqrt(ez[1]**2 + ez[2]**2))
    yaw = np.arctan2(ey[0], ex[0])

    return np.array([roll, pitch, yaw])

def conv_rpy2qt(angle_rpy:np.ndarray) -> np.ndarray:
    """ Convert angle with rpy-euler angle to quarternion angle representation """
    
    angle_qt = 0
    return angle_qt

def unit_vector(vector:np.ndarray) -> np.ndarray:
    """ Normalize vector """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def is_unit(ex, ey, ez, tol=1e-3):
    """ Check validity of unit vectors """
    
    # TODO: check function! 
    sigma = np.zeros((3,3))
    e_list = [ex, ey, ez]
    for i, ei in enumerate(e_list):
        for j, ej in enumerate(e_list):
            sigma[i, j] = np.dot(ei, ej)
    if (sigma-np.eye(3)).all()>tol :
        return False
    return True

def get_H(ex, ey, ez, d):
    """ Calculate the homogenous transformation for the given unit vectors and translation component """
    R = np.zeros((3, 3))
    for i, ei in enumerate([ex, ey, ez]):
        for j in range(3):
            R[i, j] = ei[j]
    #print(R)

    H = np.zeros((4, 4))
    H[:3, :3] = R
    H[:3, 3] = d
    H[-1, -1] = 1

    return H

def get_pose(wrist:np.ndarray, thumb:np.ndarray, index:np.ndarray) -> np.ndarray:
    """ Calculate the coordinate system pose between the fingers with euler angles """
    pose = np.zeros(shape=(6), dtype=np.float32)
    center = [0, 0, 0]
    angle = [0, 0, 0]
    
    r_it = index-thumb # vector thumb to index
    r_oc = thumb+(r_it)/2 # vector origin to tcp
    r_cw = r_oc-wrist # vector wrist to tcp
    
    center = r_oc
    # -------------------
    # create unit vectors
    ez = unit_vector(r_cw)
    ex = unit_vector(np.cross(r_it, ez))
    ey = unit_vector(np.cross(ez,ex))

    # check validity of vector
    if not is_unit(ex, ey, ez):
        raise ValueError("Unit vectors are not valid!")
    # -------------------
    # Create homogenous transformation matrix H=[[R,d][0,0,0,1]]
    H = get_H(ex,ey,ez, r_oc)

    pose[0:3] = center
    pose[3:6] = get_RPY(ex, ey, ez)

    return pose, ex, ey, ez, H

def plot_quiver(ax, v1, v2, color="black"):
    ax.quiver(v1[0], v1[1], v1[2], -v1[0]+v2[0], -v1[1]+v2[1], -v1[2]+v2[2], normalize=False, color=color)

def plot_quiver2(ax, v1, v2, color="black"):
    ax.quiver(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], length=3, normalize=True, color=color)

# Socket thread
def socket_listen():

    # Create a figure and axes
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

    #xs = ax.quiver(x, y, z, x+5, y, z, length=1, normalize=False)
    #ys = ax.quiver(x, y, z, x, y+5, z, length=1, normalize=False)
    #zs = ax.quiver(x, y, z, x, y, z+5, length=1, normalize=False)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show(block=False)
    
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    #return 
    try:
        while(True):
            wrist = np.array([10,-5,-3])
            thumb = np.array([5,5,5])
            index = np.array([0,5,0])

            pose, ex, ey, ez, H = get_pose(wrist, thumb, index)
            #pose_qt = rpy2qt(pose)
            print(pose)

            # Update the position of the point
            point_thumb._offsets3d = ([thumb[0]], [thumb[1]], [thumb[2]])
            point_wrist._offsets3d = ([wrist[0]], [wrist[1]], [wrist[2]])
            point_index._offsets3d = ([index[0]], [index[1]], [index[2]])

            point_pose._offsets3d = ([pose[0]], [pose[1]], [pose[2]])

            plot_quiver2(ax, pose[:3], ex, "red")
            plot_quiver2(ax, pose[:3], ey, "green")
            plot_quiver2(ax, pose[:3], ez, "blue")

            #xs = ax.quiver(pose[0], pose[1], pose[2], pose_x[0], pose_x[1], pose_x[2], length=3, normalize=True, color="red")
            #ys = ax.quiver(pose[0], pose[1], pose[2], pose_y[0], pose_y[1], pose_y[2], length=3, normalize=True, color="green")
            #zs = ax.quiver(pose[0], pose[1], pose[2], pose_z[0], pose_z[1], pose_z[2], length=3, normalize=True, color="blue")

            # Draw the updated plot
            plt.draw()
            plt.pause(1)
            
    except KeyboardInterrupt:
        print("Aborted!")
        plt.close()
    plt.show()
if __name__ == '__main__':
    socket_listen()