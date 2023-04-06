import subprocess
import threading
import queue
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
import numpy as np

wrist = np.array([0.0,0.0,0.0], dtype=np.float32)
thumb = np.array([0.0,0.0,0.0], dtype=np.float32)
index = np.array([0.0,0.0,0.0], dtype=np.float32)
middle = np.array([0.0,0.0,0.0], dtype=np.float32)
ring = np.array([0.0,0.0,0.0], dtype=np.float32)
pinky = np.array([0.0,0.0,0.0], dtype=np.float32)

dist = 0.0

def retrieve_output(process, output_queue):
    i=0
    while True:
        #print(f"t1->{i}")
        i += 1

        output = process.stdout.readline()
        if not output:
            break
        out = output.strip().split()
        #print(len(out))
        if len(out)!=19:
            continue
        try:
            x = float(out[0])
            y = float(out[2])
            z = float(out[1])
            global wrist
            wrist = np.array([x, y, z])

            x = float(out[3])
            y = float(out[4])
            z = float(out[5])
            global thumb
            thumb = np.array([x, y, z])

            x = float(out[6])
            y = float(out[7])
            z = float(out[8])
            global index
            index = np.array([x, y, z])

            x = float(out[9])
            y = float(out[10])
            z = float(out[11])
            global middle
            middle = np.array([x, y, z])
            
            x = float(out[12])
            y = float(out[13])
            z = float(out[14])
            global ring
            ring = np.array([x, y, z])
            
            x = float(out[15])
            y = float(out[16])
            z = float(out[17])
            global pinky
            pinky = np.array([x, y, z])      

            global dist
            dist = float(out[9])

            time.sleep(0.001)
        except Exception as e:
            print(e)

def plot_output(output_queue):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    global index
    global thumb
    global wrist
    global middle
    global ring
    global pinky

    j=0
    while True:
        #print(f"t2->{j}")
        j += 1
        plt.cla()
        ax.scatter(*wrist, c="C0", label="wrist")
        ax.scatter(*thumb, c="C1", label="thumb")
        ax.scatter(*index, c="C2", label="index")
        ax.scatter(*middle, c="C3", label="middle")
        ax.scatter(*ring, c="C4", label="ring")
        ax.scatter(*pinky, c="C5", label="pinky")

        def draw_line(point1, point2):
            ax.plot([point1[0],point2[0]], [point1[1],point2[1]], [point1[2],point2[2]], lw=1, c="black")
        
        draw_line(wrist, thumb)
        draw_line(wrist, index)
        draw_line(wrist, middle)
        draw_line(wrist, ring)
        draw_line(wrist, pinky)

        ax.set_xlim3d([-300, 300])
        ax.set_ylim3d([-300, 300])
        ax.set_zlim3d([-50, 500])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('3D Point Plot')

        ax.legend()

        plt.draw()
        plt.pause(0.001)
        
# Start the C program and redirect its stdout to a pipe
process = subprocess.Popen(["./PollingSample2"], stdout=subprocess.PIPE)

# Create a queue to hold the output data
output_queue = queue.Queue()

# Create two threads, one to retrieve the output and another to plot it
retrieve_thread = threading.Thread(target=retrieve_output, args=(process, output_queue))
plot_thread = threading.Thread(target=plot_output, args=(output_queue,))

# Start the threads
retrieve_thread.start()
plot_thread.start()

# Wait for the threads to finish
retrieve_thread.join()
plot_thread.join()
