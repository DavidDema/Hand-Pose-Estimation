import subprocess
import threading
import queue

import time
import numpy as np

import socket

HOST = 'localhost'
PORT = 8080

data = np.zeros(shape=(7,3), dtype=np.float32)
startup_time = time.time()

def socket_init():
    global data

    # send the bytes over a socket connection
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            packet_id_last = 0
            packet_time_last = 0
            while True:
                try:
                    data_send = data.flatten()
                    
                    packet_id = data_send[-2]
                    packet_time = data_send[-1]

                    if packet_id <= packet_id_last:
                        continue
                    packet_id_last = packet_id
                    conn.sendall(b'ARRAY')
                    #data_send = np.linspace(0,10, 10, dtype=np.float32)
                    print(data_send.shape, data_send[-2], data_send[-1])
                    conn.sendall(data_send.tobytes())
                except:
                #except Exception(ConnectionResetError):
                    break
                    #print("sending...")
                    #time.sleep(1)
        print('Connection closed by', addr)

def retrieve_output(process, output_queue):
    global data 
    
    wrist = np.array([0.0,0.0,0.0], dtype=np.float32)
    thumb = np.array([0.0,0.0,0.0], dtype=np.float32)
    index = np.array([0.0,0.0,0.0], dtype=np.float32)
    middle = np.array([0.0,0.0,0.0], dtype=np.float32)
    ring = np.array([0.0,0.0,0.0], dtype=np.float32)
    pinky = np.array([0.0,0.0,0.0], dtype=np.float32)
    dist = np.array([0.0,0.0,0.0], dtype=np.float32)

    i=0
    while True:

        output = process.stdout.readline()
        if not output:
            break
        out = output.strip().split()
        
        if len(out)!=19:
            continue
        try:
            x = float(out[0])
            y = float(out[2])
            z = float(out[1])
            wrist = np.array([x, y, z])

            x = float(out[3])
            y = float(out[4])
            z = float(out[5])
            thumb = np.array([x, y, z])

            x = float(out[6])
            y = float(out[7])
            z = float(out[8])
            index = np.array([x, y, z])

            x = float(out[9])
            y = float(out[10])
            z = float(out[11])
            middle = np.array([x, y, z])
            
            x = float(out[12])
            y = float(out[13])
            z = float(out[14])
            ring = np.array([x, y, z])
            
            x = float(out[15])
            y = float(out[16])
            z = float(out[17])
            pinky = np.array([x, y, z])      

            dist = np.array([float(out[9]), i, time.time()-startup_time])
            
            i += 1
            time.sleep(0.0)
        except Exception as e:
            print(e)
        data[0, :] = wrist
        data[1, :] = thumb
        data[2, :] = index
        data[3, :] = middle
        data[4, :] = ring
        data[5, :] = pinky
        data[6, :] = dist
        

# Start the C program and redirect its stdout to a pipe
process = subprocess.Popen(["./PollingSample2"], stdout=subprocess.PIPE)

# Create a queue to hold the output data
output_queue = queue.Queue()

# Create two threads, one to retrieve the output and another to plot it
retrieve_thread = threading.Thread(target=retrieve_output, args=(process, output_queue))
socket_thread = threading.Thread(target=socket_init,  args=())

# Start the threads
retrieve_thread.start()
socket_thread.start()

# Wait for the threads to finish
#retrieve_thread.join()
socket_thread.join()
#retrieve_thread.stop()
