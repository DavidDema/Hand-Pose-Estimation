import subprocess
import time
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Execute the C program and capture its stdout as a pipe
proc = subprocess.Popen(['./PollingSample2'], stdout=subprocess.PIPE)
print(proc)

# create a 3D figure
fig = plt.figure()
ax = plt.axes(projection='3d')

# Read the output one line at a time
while True:
    output_line = proc.stdout.readline().decode('utf-8')
    if not output_line:
        break

    out = output_line.strip().split()
    
    try:
        wrist = np.array([float(out[0]), float(out[1]), float(out[2])], dtype=np.float32)
        dist = np.array([float(out[3])], dtype=np.float32)
        
        #print(index, thumb, wrist)
        #v_index[0], v_index[1], v_index[2], v_thumb[0], v_thumb[1], v_thumb[2], v_wrist[0], v_wrist[1], v_wrist[2], a = map(float, output_line.strip().split())
        
        plt.cla()
        # plot a point in 3D space
        ax.scatter(index[0], index[1], index[2], color='red')
        ax.scatter(thumb[0], thumb[1], thumb[2], color='blue')
        ax.scatter(wrist[0], wrist[1], wrist[2], color='green')

        # set axis labels and title
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_zlim(-200, 500)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('3D Point Plot')
        
        print(wrist)
        # show the plot
        plt.pause(0.01)

    except Exception as e:
        print(e)
        