#!/usr/bin/env python3

import threading

import socket
import numpy as np
import time

import rospy
from geometry_msgs.msg import Pose

# For socket communication
HOST = 'localhost'  # The server's hostname or IP address
PORT = 8080        # The port used by the server

# For ROS node
rospy.init_node('hpe_leap')
rate = rospy.Rate(100) # Rate in Hz

pub_pose = rospy.Publisher('/hpe_leap/pose', Pose, queue_size=10, tcp_nodelay=True)
pose_msg = Pose()

def rpy2qt(pose_rpy):
    """ Convert pose with rpy-euler angle to quarternion angle representation """
    pose_qt = np.zeros(shape=(7), dtype=np.float32)
    
    # TODO convert
    pose_qt[0:6] = pose_rpy

    return pose_qt

def get_pose(wrist:np.ndarray, thumb:np.ndarray, index:np.ndarray) -> np.ndarray:
    """ Calculate the coordinate system pose between the fingers with euler angles """
    pose = np.zeros(shape=(6), dtype=np.float32)
    
    pose[0:3] = (thumb-index)

    # TODO add angle
    # pose[3:6] = [0, 0, 0]

    return pose

# Socket thread
def socket_listen():
    global pose_msg

    wrist = np.zeros(shape=(3), dtype=np.float32)
    thumb = np.zeros(shape=(3), dtype=np.float32)
    index = np.zeros(shape=(3), dtype=np.float32)
    middle = np.zeros(shape=(3), dtype=np.float32)
    ring = np.zeros(shape=(3), dtype=np.float32)
    pinky = np.zeros(shape=(3), dtype=np.float32)
    dist = np.zeros(shape=(3), dtype=np.float32)
    
    # create socket connection to read data from Ultraleap 3Di
    fps = 0.0
    fps_list = np.zeros(1000)
    logging_timer = time.time()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            rospy.loginfo("Socket connected.")
            while True:
                t1 = time.time()
                data = s.recv(1024)
                if not data:
                    break

                try:
                    received_arr = np.frombuffer(data, dtype=np.float32)
                    if received_arr.shape[0] > 43:
                        continue
                    wrist = received_arr[(0*3):(0*3+3-1)] #0:3
                    thumb = received_arr[(1*3):(1*3+3-1)] #3:6
                    index = received_arr[(2*3):(2*3+3-1)] #6:9
                    #print('Received', repr(received_arr), received_arr.shape)
                    
                    pose_rpy = get_pose(wrist, thumb, index)
                    pose_qt = rpy2qt(pose_rpy)

                    pose_msg.position.x = pose_qt[0]
                    pose_msg.position.y = pose_qt[1]
                    pose_msg.position.z = pose_qt[2]
                    pose_msg.orientation.x = pose_qt[3]
                    pose_msg.orientation.y = pose_qt[4]
                    pose_msg.orientation.z = pose_qt[5]
                    pose_msg.orientation.w = pose_qt[6]

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

                
    except Exception as e:
        print(e)
        # Loop for testing without socket connection with Ultraleap
        rospy.loginfo("No socket connection - Fixed pose !")

        pose_rpy = np.zeros(shape=(6), dtype=np.float32)
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
    socket_thread = threading.Thread(target=socket_listen,  args=())

    # Start the threads
    socket_thread.start()
    ros_thread.start()

    # Wait for the threads to finish
    #retrieve_thread.join()
    socket_thread.join()
    ros_thread.join()