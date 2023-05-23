#!/usr/bin/env python3

import threading

import socket
import numpy as np
import time

import matplotlib.pyplot as plt

import tf.transformations as tft

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
    pose_qt = np.zeros(shape=(7), dtype=np.float32)
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
    ey = unit_vector(np.cross(ex,ez))
    
    # check validity of vector
    if not is_unit(ex, ey, ez):
        raise ValueError("Unit vectors are not valid!")
    # -------------------
    # Create homogenous transformation matrix H=[[R,d][0,0,0,1]]
    H = get_H(ex,ey,ez, r_oc)
    #H = np.zeros((4,4))

    pose[0:3] = center
    pose[3:6] = get_RPY(ex, ey, ez)

    pose_qt[0:3] = center
    pose_qt[3:7] = tft.quaternion_from_euler(pose[3], pose[4], pose[5])

    return pose, pose_qt, ex, ey, ez, H

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

# Socket thread
def socket_listen():
    global pose
    global pose_qt

    global wrist
    global thumb
    global index

    middle = np.zeros(shape=(3), dtype=np.float32)
    ring = np.zeros(shape=(3), dtype=np.float32)
    pinky = np.zeros(shape=(3), dtype=np.float32)
    dist = np.zeros(shape=(3), dtype=np.float32)
    
    # create socket connection to read data from Ultraleap 3Di
    fps = 0.0
    fps_list = np.zeros(1000)
    logging_timer = time.time()

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
    #ax.set_xlim([-10, 10])
    #ax.set_ylim([-10, 10])
    #ax.set_zlim([-10, 10])

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
                    print('Received', repr(received_arr), received_arr.shape)

                    wrist = received_arr[0:3] #0:3
                    thumb = received_arr[3:6] #3:6
                    index = received_arr[6:9] #6:9
                    
                    pose, pose_qt, ex, ey, ez, H = get_pose(wrist, thumb, index)

                    # Update the position of the point
                    point_thumb._offsets3d = ([thumb[0]], [thumb[1]], [thumb[2]])
                    point_wrist._offsets3d = ([wrist[0]], [wrist[1]], [wrist[2]])
                    point_index._offsets3d = ([index[0]], [index[1]], [index[2]])

                    point_pose._offsets3d = ([pose[0]], [pose[1]], [pose[2]])
                    
                    plot_units(ax, ex, ey, ez, H[:3, 3])

                    # Draw the updated plot
                    plt.draw()
                    plt.pause(0.1)

                    t2 = time.time()
                    fps = 1/(t2-t1)
                except Exception as e:
                    print(e)
                fps_list = np.roll(fps_list, -1)
                fps_list[-1] = fps

                if logging_timer>1:    
                    pass
                    #print('FPS:%f(%.2f/%.2f)'%(np.mean(fps_list),fps,np.cov(fps_list)))
                    if len(np.where(fps_list==0.0)[0])>10:
                        fps_print = fps
                    else:
                        fps_print = np.mean(fps_list)
                        #rospy.loginfo('FPS:%.0f'%(fps_print))

    except Exception as e:
        print(e)
        # Loop for testing without socket connection with Ultraleap
        #rospy.loginfo("No socket connection - Fixed pose !")

        while(True):
            wrist = np.array([0,0,0])
            thumb = np.array([5,-5,-3])
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


            #plot_quiver(ax, wrist, thumb)
            #plot_quiver(ax, wrist, index)

            # Draw the updated plot
            plt.draw()
            plt.pause(0.1)
            

if __name__ == '__main__':
    
    if THREADING:
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
    else:
        socket_listen()