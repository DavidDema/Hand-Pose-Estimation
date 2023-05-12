#!/usr/bin/env python3

import threading

import socket
import numpy as np
import time

import rospy
from std_msgs.msg import Float32

# For socket communication
HOST = 'localhost'  # The server's hostname or IP address
PORT = 8080        # The port used by the server

# For ROS node
rospy.init_node('hpe_leap')
rate = rospy.Rate(100)

pub_index = rospy.Publisher('/hpe_leap/index', Float32, queue_size=10, tcp_nodelay=True)
pub_thumb = rospy.Publisher('/hpe_leap/thumb', Float32, queue_size=10, tcp_nodelay=True)
pub_wrist = rospy.Publisher('/hpe_leap/wrist', Float32, queue_size=10, tcp_nodelay=True)

wrist = np.array([0.0,0.0,0.0], dtype=np.float32)
thumb = np.array([0.0,0.0,0.0], dtype=np.float32)
index = np.array([0.0,0.0,0.0], dtype=np.float32)
middle = np.array([0.0,0.0,0.0], dtype=np.float32)
ring = np.array([0.0,0.0,0.0], dtype=np.float32)
pinky = np.array([0.0,0.0,0.0], dtype=np.float32)
dist = np.array([0.0,0.0,0.0], dtype=np.float32)

# Socket thread
def socket_listen():
    global index
    global thumb
    global wrist

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        fps = np.zeros(100)
        num_packets = 0
        num_success = 0
        while True:
            t1 = time.time()
            
            data = s.recv(1024)
            num_packets += 1
            if not data:
                break
            if data == b'ARRAY':
                data = s.recv(1024)
                try:
                    received_arr = np.frombuffer(data, dtype=np.float32)
                    if received_arr.shape[0] > 43:
                        continue
                    wrist = received_arr[(0*3):(0*3+3-1)]
                    thumb = received_arr[(1*3):(1*3+3-1)]
                    index = received_arr[(2*3):(2*3+3-1)]

                    #print('Received', repr(received_arr), received_arr.shape)

                    num_success += 1

                    t2 = time.time()
                    fps[0:-2] = fps[1:-1]
                    fps[-1] = 1/(t2-t1)
                    #print('FPS:', np.mean(fps), "\nLoss:", num_success/num_packets)
                except Exception as e:
                    pass
                    #print(e)
                    #print(data.decode())
                
            else:
                pass
                # process the message (in this case, print it)
                #message = data.decode()
                #print('Received message:')
                #print(message)

def ros_publish():
    global index
    global thumb
    global wrist

    while not rospy.is_shutdown():
        #value = 42 # change this value to the integer you want to publish
        rospy.loginfo("Publishing!")
        pub_index.publish(index[0])
        pub_thumb.publish(thumb[0])
        pub_wrist.publish(wrist[0])
        rate.sleep()

# Create two threads, one to retrieve the output and another to plot it
ros_thread = threading.Thread(target=ros_publish, args=())
socket_thread = threading.Thread(target=socket_listen,  args=())

# Start the threads
socket_thread.start()
ros_thread.start()

# Wait for the threads to finish
#retrieve_thread.join()
socket_thread.join()
ros_thread.start()