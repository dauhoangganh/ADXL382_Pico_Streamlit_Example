import serial
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time
import sys
import queue
from .utils import time_plot, twos_complement_to_decimal, SAMPLE_RATE, DURATION

data_queue = queue.Queue()

# === Data Buffers ===
MAX_SAMPLES = 32000  # How many samples to keep in the plot
x_vals = deque(maxlen=MAX_SAMPLES)
y_vals = deque(maxlen=MAX_SAMPLES)
z_vals = deque(maxlen=MAX_SAMPLES)
# samples_x = deque(maxlen=MAX_SAMPLES)
# samples_y = deque(maxlen=MAX_SAMPLES)
# samples_z = deque(maxlen=MAX_SAMPLES)


# sample_count_x = 0
# sample_count_y = 0
# sample_count_z = 0
# === Regex Pattern to Parse Lines ===
pattern = re.compile(r'([xyz])? ?([\dA-Fa-f]+)')
# pattern = re.compile(r'(0\d): ?(\d+)')
idle_timeout = 5

# Shared state with a lock
# progress_state = {"value": 0.0}

def read_serial_data(ser, x_vals_q, y_vals_q, z_vals_q, progress_state, done_event, messages_q):
    last_data_time = time.time()
    try:
        while True:
            if ser.in_waiting:
                line = ser.read_until(b"\n").decode(errors='ignore').strip()
                last_data_time = time.time()
                matches = pattern.findall(line)
                if len(matches) == 1:
                    # print(line)
                    axis, value = matches[0]
                    # value = float(value)
                    value = round(twos_complement_to_decimal(value) * 500e-6, 4)  # Convert to g value by scaling with ADXL382 sacle factor
                    if axis == "x":
                        x_vals.append(value)
                    elif axis == "y":
                        y_vals.append(value)
                    elif axis == "z":
                        z_vals.append(value)
                    # Update progress every 10% or when max samples reached
                    if len(x_vals) % (SAMPLE_RATE*DURATION/10) == 0 or len(x_vals) == MAX_SAMPLES: 
                        progress_state["value"] = min(100.0, len(x_vals) / MAX_SAMPLES * 100.0)
                    
                # else:
                #     messages_q.put(line)
                       
                if line == "End of while loop":
                    messages_q.put("End of data stream received. Exiting...")
                    progress_state["value"] = min(100.0, len(x_vals) / MAX_SAMPLES * 100.0)
                    break
            else:
                if (time.time() - last_data_time > idle_timeout):
                    messages_q.put("No data received for a while. Assuming end of stream.")
                    progress_state["value"] = min(100.0, len(x_vals) / MAX_SAMPLES * 100.0)
                    break
            
    except serial.SerialException:
        # print("Serial port disconnected. Exiting...")
        messages_q.put("Serial port disconnected. Exiting...")
        ser.close()
        sys.exit()
    x_vals_q.put(list(x_vals))
    y_vals_q.put(list(y_vals))
    z_vals_q.put(list(z_vals))
    # print("xcount:", len(x_vals), "ycount:", len(y_vals), "zcount:", len(z_vals))
    done_event.set()
    return x_vals, y_vals, z_vals




