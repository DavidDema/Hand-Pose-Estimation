#!/usr/bin/env python3
# ipc_client.py

import socket
import numpy as np
import time

HOST = 'localhost'  # The server's hostname or IP address
PORT = 8080        # The port used by the server

fps_list = np.zeros(1000)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    fps = 0.0
    while True:
        t1 = time.time()
        
        data = s.recv(1024)
        if not data:
            break
        try:
            received_arr = np.frombuffer(data, dtype=np.float32)
            if received_arr.shape[0] > 43:
                print('Wrong received', repr(received_arr), received_arr.shape)
                continue
            print('Received', repr(received_arr), received_arr.shape)
            
            t2 = time.time()
            fps = 1/(t2-t1)
        except Exception as e:
            print(e)
            
        fps_list = np.roll(fps_list, -1)
        fps_list[-1] = fps

        #print('FPS:%f(%.2f/%.2f)'%(np.mean(fps_list),fps,np.cov(fps_list)))
        if len(np.where(fps_list==0.0)[0])>10:
            fps_print = fps
        else:
            fps_print = np.mean(fps_list)
        print('FPS:%.0f'%(fps_print))