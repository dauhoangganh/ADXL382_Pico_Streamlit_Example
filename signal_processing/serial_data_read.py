import serial
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time
import sys
from .utils import SAMPLE_RATE, DURATION

# data_queue = queue.Queue()

# === Data Buffers ===
MAX_SAMPLES = SAMPLE_RATE*DURATION  # How many samples to keep in the plot

# === Regex Pattern to Parse Lines ===
# pattern = re.compile(rb'(0\d)([0-9A-Fa-f]{4})')
# pattern = re.compile(r'(0\d): ?(\d+)')
idle_timeout = 1

def read_serial_data(ser, x_vals_q, y_vals_q, z_vals_q, progress_state, done_event, messages_q):
    last_data_time = time.time()
    ser.write(b"START\n")
    try:
        #reset x_vals, y_vals, z_vals
        x_vals = []
        y_vals = []
        z_vals = []
        progress_bytes = 0
        byte_buffer = bytearray()
        # ser.reset_input_buffer() 
        # ser.reset_output_buffer()
        t = time.time()
        while not done_event.is_set():
            
            if ser.in_waiting:
                line = ser.read(1024*2)  # read a line from the serial port
                last_data_time = time.time()
                if not line:
                    continue
                else:
                    # print(line)
                    byte_buffer.extend(line)  # append the line to the byte buffer
                    if len(byte_buffer) - progress_bytes >= 300000:
                        progress_state["value"] = min(100.0, len(byte_buffer)/ (1536000) * 100.0)
                        progress_bytes += 100000  # update the progress state every 100000 bytes
                    if b"End" in line:
                        messages_q.put("End of data stream received.")
                        progress_state["value"] = 100.0
                        break
            else:
                if (time.time() - last_data_time > idle_timeout):
                    messages_q.put("No data received for a while. Assuming end of stream.")
                    progress_state["value"] = min(100.0, len(byte_buffer)/ (768000*2) * 100.0)
                    break
            
    except serial.SerialException:
        # print("Serial port disconnected. Exiting...")
        messages_q.put("Serial port disconnected. Exiting...")
        ser.close()
        sys.exit()
    
    print(f"Read {len(byte_buffer)} lines from serial port.")
    # print(byte_buffer[:-200])
    byte_buffer = byte_buffer.split(b'\r\n')  # split the byte buffer into lines
    for line in byte_buffer:
        # print(line)
        if len(line) == 6:
            try:
                tag = bytes(line[0:2])
                hex_value = bytes(line[2:6])

                if tag == b'00':
                    x_vals.append(hex_value) # scale by scaling factor of adxl382
                elif tag == b'01':
                    y_vals.append(hex_value) # scale by scaling factor of adxl382
                elif tag == b'02':
                    z_vals.append(hex_value) # scale by scaling factor of adxl382
            except Exception as e:
                continue  # skip corrupt line
    
    x_vals_q.put(x_vals)
    y_vals_q.put(y_vals)
    z_vals_q.put(z_vals)
    print("Time taken to read data:", time.time() - t)
    done_event.set()
    #remove the byte_buffer
    del byte_buffer, x_vals, y_vals, z_vals
    ser.close()
    print("Time taken to read data:", time.time() - t)
    # return x_vals, y_vals, z_vals




