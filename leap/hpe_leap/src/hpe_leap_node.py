#!/usr/bin/env python3

import socket
import numpy as np
import time

import rospy
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker

from scipy.spatial.transform import Rotation
#import tf.transformations as tft

TRACKING_MODE = "H"
#TRACKING_MODE = "D"
UNIT_SCALING = 1e3 # millimeters

# For socket communication
HOST = 'localhost'  # The server's hostname or IP address
PORT = 8080        # The port used by the server

# For ROS node
rospy.init_node('hpe_leap')
rate = rospy.Rate(100) # Rate in Hz

pub_pose = rospy.Publisher('/hpe_leap/pose', Pose, queue_size=10, tcp_nodelay=True)
pose_msg = Pose()

def create_rviz_marker(name:str, type:int=2, color=[1.0, 0.0, 0.0], scale=[0.05, 0.05, 0.05], brightness=1.0):
    marker_pub = rospy.Publisher(f"/hpe_leap/vis_marker_{name}", Marker, queue_size = 2)
    marker = Marker()

    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    marker.type = type
    marker.id = 0
    # Set the scale of the marker
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    # Set the color
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = brightness

    return marker, marker_pub

#marker_c, marker_c_pub = create_rviz_marker("c", type=0, color=[1.0, 0.0, 0.0], scale=[0.1, 0.01, 0.01])
marker_c, marker_c_pub = create_rviz_marker("c", type=2, color=[1.0, 1.0, 1.0])

marker_w, marker_w_pub = create_rviz_marker("w", type=2, color=[1.0, 1.0, 1.0])
marker_i, marker_i_pub = create_rviz_marker("i", type=2, color=[1.0, 1.0, 1.0], brightness=0.5)
marker_t, marker_t_pub = create_rviz_marker("t", type=2, color=[1.0, 1.0, 1.0], brightness=0.5)

marker_ex, marker_ex_pub = create_rviz_marker("ex", type=2, color=[1.0, 0.0, 0.0])
marker_ey, marker_ey_pub = create_rviz_marker("ey", type=2, color=[0.0, 1.0, 0.0])
marker_ez, marker_ez_pub = create_rviz_marker("ez", type=2, color=[0.0, 0.0, 1.0])

def unit_vector(vector:np.ndarray) -> np.ndarray:
    """ Normalize vector """
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return np.array(vector / norm)

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

def get_H(degz:float=0.0, degy:float=0.0, degx:float=0.0, d:np.ndarray=np.zeros(3)) -> np.ndarray:
    """ Create Homogeneous Transformation Matrix with given rotation euler angles in degrees (zyx-rotation) and translation component d"""
    R = Rotation.from_euler("zyx", [degz, degy, degx], degrees=True).as_matrix()
    H = np.zeros((4, 4))
    H[:3, :3] = R
    H[:3, 3] = d
    H[-1, -1] = 1
    return H

def tf_vector(vector:np.ndarray, H:np.ndarray) -> np.ndarray:
    """ Transform vector using a transformation Matrix H (4x4; [[R,d],[0,1]])"""
    v2 = np.zeros(shape=(len(vector)+1))
    v2[:3] = vector
    v2[-1] = 1
    return (H @ v2)[:3]

def get_pose(wrist:np.ndarray, thumb:np.ndarray, index:np.ndarray) -> np.ndarray:
    """ Calculate the coordinate system pose between the fingers with quaternion angles """
    pose = np.zeros(shape=(7), dtype=np.float32)
    center = [0, 0, 0]
    
    # Coordinate system 1 is origin of camera - 0 is origin of robot
    r_it = index-thumb # vector thumb to index
    r_1c = thumb+(r_it)/2 # vector origin to tcp
    r_cw = r_1c-wrist # vector wrist to tcp
    
    center = r_1c
    # -------------------
    # create unit vectors
    ez = unit_vector(r_cw)
    ex = unit_vector(np.cross(r_it, ez))
    ey = unit_vector(np.cross(ex,ez))
    
    # check validity of vector
    if not is_unit(ex, ey, ez):
        raise ValueError("Unit vectors are not valid!")
    # -------------------

    # Calculate pose with quaternion angle repr.
    pose[0:3] = center

    R = np.zeros((3,3))
    for i, ei in enumerate([ex, ey, ez]):
        for j in range(3):
            R[j, i] = ei[j]
    r = Rotation.from_matrix(R)
    pose[3:7] = r.as_quat()
    #print(r.as_matrix())

    # Create homogenous transformation matrix H=[[R,d][0,0,0,1]]
    H = np.zeros((4,4))
    H[:3, :3] = R
    H[:3, 3] = center
    H[-1, -1] = 1
    return pose, H

def main():
    # create socket connection to read data from Ultraleap 3Di
    fps = 0.0
    fps_list = np.zeros(1000)
    logging_timer = time.time()

    # Convert location of Leap camera to origin
    # Unit in millimeters
    # Leap (facing the camera directly)
    # x (+right, -left)
    # y (+back -front) back -> moving away from camera
    # z (+down -up)

    # Desktop mount
    # right = x+
    # left = x-
    # back = z+
    # front = z-
    # up = y+
    # down = y-
    H_desktop = get_H(degx=90, degy=0, degz=0, d=np.zeros(3))

    # Head/Robot mount
    # right = x-
    # left =  x+
    # back = z+
    # front = z-
    # up = y-
    # down = y+
    H_head = get_H(degx=-270, degy=0, degz=180, d=np.array([0,-100.0, +200.0]))

    if TRACKING_MODE == "D":
        rospy.loginfo("Ultraleap in Desktop mode.")
        H_tf = H_desktop
    elif TRACKING_MODE == "H":
        rospy.loginfo("Ultraleap in Head/Robot mounted mode.")
        H_tf = H_head
    else:
        raise ValueError("TRACKING MODE invalid !")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            rospy.loginfo("Socket connected.")
            while not rospy.is_shutdown():
                t1 = time.time()
                data = s.recv(1024)
                if not data:
                    break
                try:
                    received_arr = np.frombuffer(data, dtype=np.float32)
                    if received_arr.shape[0] > 43:
                        continue
                    wrist = received_arr[0:3]
                    thumb = received_arr[3:6]
                    index = received_arr[6:9]
                    
                    if True:
                    # Better for visualization 
                        UNIT_SCALING = 1e2

                    thumb = tf_vector(thumb, H_tf)/UNIT_SCALING
                    index = tf_vector(index, H_tf)/UNIT_SCALING
                    wrist = tf_vector(wrist, H_tf)/UNIT_SCALING
                    pose, H = get_pose(wrist, thumb, index)

                    
                    def set_marker_pose(marker, center=[0,0,0], orientation=[0,0,0,1]):
                        marker.pose.position.x = center[0]
                        marker.pose.position.y = center[1]
                        marker.pose.position.z = center[2]
                        marker.pose.orientation.x = orientation[0]
                        marker.pose.orientation.y = orientation[1]
                        marker.pose.orientation.z = orientation[2]
                        marker.pose.orientation.w = orientation[3]

                    # Pose message
                    pose_msg.position.x = pose[0]
                    pose_msg.position.y = pose[1]
                    pose_msg.position.z = pose[2]
                    pose_msg.orientation.x = pose[3]
                    pose_msg.orientation.y = pose[4]
                    pose_msg.orientation.z = pose[5]
                    pose_msg.orientation.w = pose[6]
                    pub_pose.publish(pose_msg)

                    # Centerpoint visulization
                    #set_marker_pose(marker_c, pose[:3], pose[3:7])
                    set_marker_pose(marker_c, pose[:3])

                    # Hand landmark visulization
                    set_marker_pose(marker_t, thumb)
                    set_marker_pose(marker_i, index)
                    set_marker_pose(marker_w, wrist)
                    
                    # Unit vector visulization
                    scale = 0.1
                    set_marker_pose(marker_ex, tf_vector(np.array([1, 0, 0])*scale, H))
                    set_marker_pose(marker_ey, tf_vector(np.array([0, 1, 0])*scale, H))
                    set_marker_pose(marker_ez, tf_vector(np.array([0, 0, 1])*scale, H))

                    # Publish all the marker messages
                    marker_c_pub.publish(marker_c)

                    marker_t_pub.publish(marker_t)
                    marker_i_pub.publish(marker_i)
                    marker_w_pub.publish(marker_w)

                    marker_ex_pub.publish(marker_ex)
                    marker_ey_pub.publish(marker_ey)
                    marker_ez_pub.publish(marker_ez)

                    # Get time passed and calculate FPS
                    t2 = time.time()
                    fps = 1/(t2-t1)
                except Exception as e:
                    print(e)
                fps_list = np.roll(fps_list, -1)
                fps_list[-1] = fps

                if logging_timer>1:    
                    #print('FPS:%f(%.2f/%.2f)'%(np.mean(fps_list),fps,np.cov(fps_list)))
                    if len(np.where(fps_list==0.0)[0])>10:
                        fps_print = fps
                    else:
                        fps_print = np.mean(fps_list)
                        rospy.loginfo('FPS:%.0f'%(fps_print))
                rate.sleep()
                
    except Exception as e:
        print(e)
        # Loop for testing without socket connection with Ultraleap
        rospy.loginfo("No socket connection - Fixed pose !")

        # Insert pseudo landmarks
        wrist = np.array([0,0,0])
        thumb = np.array([5,-5,-3])
        index = np.array([0,5,0])

        pose, H = get_pose(wrist, thumb, index)

        pose_msg.position.x = pose[0]
        pose_msg.position.y = pose[1]
        pose_msg.position.z = pose[2]
        pose_msg.orientation.x = pose[3]
        pose_msg.orientation.y = pose[4]
        pose_msg.orientation.z = pose[5]
        pose_msg.orientation.w = pose[6]
        
        pub_pose.publish(pose_msg)
        rate.sleep()

if __name__ == '__main__':
    main()