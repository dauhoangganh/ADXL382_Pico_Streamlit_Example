import serial
import re
import matplotlib.pyplot as plt
from collections import deque
import time
import sys
from signal_processing.time_plot import read_serial_data
from signal_processing.fft_plot import plot_fft_data
from signal_processing.utils import time_plot, SAMPLE_RATE, DURATION
import streamlit as st
import threading
import queue
from threading import Thread, Event
import numpy as np
import streamlit.components.v1 as components
# samples_x_q = queue.Queue()
# samples_y_q = queue.Queue()
# samples_z_q = queue.Queue()
x_vals_q= queue.Queue()
y_vals_q = queue.Queue()
z_vals_q = queue.Queue()
messages_q = queue.Queue()
progress_state = {"value": 0.01}
done_event = Event()
# === Serial Configuration ===
SERIAL_PORT = 'COM6'        # Change this to your serial port
BAUD_RATE = 4000000       # Match your device's baud rate

# Function to wait for the COM port to become available
def wait_for_com_port(SERI, BAUD_RATE, timeout):
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=timeout)
            # Flush any old input that might be sitting in the buffer
            ser.flushInput()
            # Optional: You can also flush output (if applicable)
            ser.flushOutput()
            # print(f"Connected to {SERIAL_PORT}")
            st.toast(f"Connected to {SERIAL_PORT}", icon="🆗")
            return ser
        except serial.SerialException:
            # print(f"Waiting for {SERIAL_PORT} to become available...")
            st.toast(f"Waiting for {SERIAL_PORT} to become available...", icon="🔍")
            time.sleep(1)

def background_serial_read():
    read_serial_data(ser, x_vals_q, y_vals_q, z_vals_q, progress_state, done_event, messages_q)

if __name__ == "__main__":
    
    
    st.set_page_config(
    page_title="Motor fault detection",
    page_icon="🏂",
    layout="wide"
    )
    
    start_button = st.button("Start Data Acquisition")
    progress_bar = st.empty()
    col = st.columns((2,2,1), gap='medium')
    with col[0]:
        st.markdown("<h3 style='text-align: center;'>X, Y, Z gee values</h3>", unsafe_allow_html=True)
        plot_placeholder = st.empty()
        
    with col[1]:
        st.markdown("<h3 style='text-align: center;'>Spectrum graph</h3>", unsafe_allow_html=True)
        fft_placeholder = st.empty()
    with col[2]:
        st.markdown("<h3 style='text-align: center;'>Fault Detection</h3>", unsafe_allow_html=True)
        components.html("""
            <html>
            <head>
            <style>
                body {
                font-family: sans-serif;
                }
                .container {
                max-width: 400px;
                margin: auto;
                }
                .box {
                background-color: white;
                border-radius: 1rem;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                padding: 1rem;
                margin-bottom: 1rem;
                transition: box-shadow 0.3s ease;
                }
                .box:hover {
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                }
                .text {
                color: #1f2937;
                font-size: 1.125rem;
                font-weight: 500;
                }
            </style>
            </head>
            <body>
            <div class="container">
                <div class="box"><p class="text">Normal state</p></div>
                <div class="box"><p class="text">Fault 1</p></div>
                <div class="box"><p class="text">Fault 2</p></div>
                <div class="box"><p class="text">Fault 3</p></div>
                <div class="box"><p class="text">Fault 4</p></div>
            </div>
            </body>
            </html>
    """, height=600)

    if start_button:
        ser = wait_for_com_port(SERIAL_PORT, BAUD_RATE, timeout=1) #change COM number according to your PC
        # Start the background thread
        thread = threading.Thread(target=background_serial_read)
        thread.start()

        # try:
        #     samples_x, samples_y, samples_z, x_vals, y_vals, z_vals = read_serial_data(ser, my_bar)
        #     with col[0]:
        #         fig = plot_data(samples_x, samples_y, samples_z, x_vals, y_vals, z_vals)
        #         plot_placeholder.pyplot(fig)
        # except:
        #     ser.close()
        #     st.error("Error reading data from the serial port. Please check the connection and try again.")  
        #     import traceback
        #     print(traceback.format_exc())
        while not done_event.is_set():
            progress_bar.progress(progress_state["value"]/100, text=f"Reading sensor data... {progress_state['value']:.1f}%")
            if not messages_q.empty():
                message = messages_q.get()
                st.toast(message, icon="ℹ️")
            time.sleep(0.2)
        if done_event.is_set():
            # Store data in session_state
            st.session_state['x_vals'] = x_vals_q.get()
            st.session_state['y_vals'] = y_vals_q.get()
            st.session_state['z_vals'] = z_vals_q.get()
            progress_bar.progress(100, text="Finished reading sensor data. Plotting...")
            with col[0]:
                fig = time_plot([st.session_state['x_vals'], st.session_state['y_vals'], st.session_state['z_vals']], 0 , len(st.session_state['z_vals']), "time_plot")
                plot_placeholder.pyplot(fig)
                
            with col[1]:
                if not len(st.session_state['x_vals'])==32000 or not len(st.session_state['y_vals'])==32000 or not len(st.session_state['z_vals'])==32000:
                    st.error("Sample length is not 32000. Please check the data acquisition process.")
                    st.error("x: " + str(len(st.session_state['x_vals'])) + ", y: " + str(len(st.session_state['y_vals'])) + ", z: " + str(len(st.session_state['z_vals'])))
                else:
                    fft_fig, len_xf = plot_fft_data(st.session_state['x_vals'], st.session_state['y_vals'], st.session_state['z_vals'])
                    fft_placeholder.pyplot(fft_fig)
                    st.session_state['fft_fig'] = fft_fig
                    st.session_state['len_xf'] = len_xf
        
    if 'x_vals' in st.session_state:
        x_vals = st.session_state['x_vals']
        y_vals = st.session_state['y_vals']
        z_vals = st.session_state['z_vals']

        # Sliders for X axis range (time index)
        with col[0]:
            x_start, x_end = st.slider(
                "Select X-axis range",
                0, len(x_vals), (0, len(x_vals)), step=200
            )

            # Sliders for Y axis range (value range)
            y_min_plot_1 = min(x_vals)
            y_min_plot_2 = min(y_vals)
            y_min_plot_3 = min(z_vals)
            y_max_plot_1 = max(x_vals)
            y_max_plot_2 = max(y_vals)
            y_max_plot_3 = max(z_vals)
            y_range_plot_1 = st.slider(
                "Select Y-axis range for acceleration X-axis plot",
                float(y_min_plot_1), float(y_max_plot_1), (float(y_min_plot_1), float(y_max_plot_1))
            )
            y_range_plot_2 = st.slider(
                "Select Y-axis range for acceleration Y-axis plot",
                float(y_min_plot_2), float(y_max_plot_2), (float(y_min_plot_2), float(y_max_plot_2))
            )
            y_range_plot_3 = st.slider(
                "Select Y-axis range for acceleration Z-axis plot",
                float(y_min_plot_3), float(y_max_plot_3), (float(y_min_plot_3), float(y_max_plot_3))
            )

            fig = time_plot([x_vals, y_vals, z_vals], x_start, x_end, "time_plot", y_range=[y_range_plot_1, y_range_plot_2, y_range_plot_3])
            plot_placeholder.pyplot(fig)
        if len(x_vals) == SAMPLE_RATE*DURATION and len(y_vals) == SAMPLE_RATE*DURATION and len(z_vals) == SAMPLE_RATE*DURATION:
            with col[1]:
                # fft_fig = plot_fft_data(st.session_state['x_vals'], st.session_state['y_vals'], st.session_state['z_vals'])
                fft_placeholder.pyplot(st.session_state['fft_fig'])
                x_start_fft, x_end_fft = st.slider(
                    "Select X-axis range",
                    0, int(st.session_state['len_xf']/2), (0, int(st.session_state['len_xf']/2)), step=100
                )
                fft_fig, len_xf = plot_fft_data(st.session_state['x_vals'], st.session_state['y_vals'], st.session_state['z_vals'], start=x_start_fft*2, stop=x_end_fft*2)
                fft_placeholder.pyplot(fft_fig)

                save_button = st.button("Save data to .csv")
                if save_button:
                    filename = st.text_input("Enter CSV file name (without .csv)", value="100_normal")

                    if filename:  # Only show the download/save once a filename is entered
                        import pandas as pd
                        df = pd.DataFrame({
                            'X': st.session_state['x_vals'],
                            'Y': st.session_state['y_vals'],
                            'Z': st.session_state['z_vals']
                        })
                        csv_filename = filename + ".csv"
                        df.to_csv(csv_filename, index=False)
                        st.success(f"Data saved to {csv_filename}")
                
