import serial
import re
import matplotlib.pyplot as plt
from collections import deque
import time
import sys
from signal_processing.serial_data_read import read_serial_data
from signal_processing.fft_plot import plot_fft_data
from signal_processing.utils import hex_strings_to_int_array, time_plot, SAMPLE_RATE, DURATION, RESAMPLE_RATE, twos_complement_to_decimal_array, _downSampler,  apply_wiener_filter, _dataScaler
import streamlit as st
import threading
import queue
from threading import Thread, Event
import numpy as np
import streamlit.components.v1 as components
import pandas as pd
from tensorflow.keras.models import load_model
from pathlib import Path
import joblib

x_vals_q= queue.Queue()
y_vals_q = queue.Queue()
z_vals_q = queue.Queue()
messages_q = queue.Queue()
progress_state = {"value": 0.0}
done_event = Event()
serial_thread = None    
# === Serial Configuration ===
SERIAL_PORT = 'COM6'        # Change this to your serial port
BAUD_RATE = 2000000      # Match your device's baud rate

# Function to wait for the COM port to become available
def wait_for_com_port(SERIAL_PORT, BAUD_RATE, timeout):
    while True:
        try:
            # set_serial_buffer_windows(SERIAL_PORT)
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=timeout)
            # print(f"Connected to {SERIAL_PORT}")
            st.toast(f"Connected to {SERIAL_PORT}", icon="🆗")
            return ser
        except serial.SerialException:
            # print(f"Waiting for {SERIAL_PORT} to become available...")
            st.toast(f"Waiting for {SERIAL_PORT} to become available...", icon="🔍")
            time.sleep(1)

def background_serial_read():
    read_serial_data(ser, x_vals_q, y_vals_q, z_vals_q, progress_state, done_event, messages_q)

@st.cache_data
def load_data():
    return pd.read_csv("output_data/data.csv", header=0, names=['X', 'Y', 'Z'], dtype={'X': float, 'Y': float, 'Z': float})

def render_fault_ui(placeholder=None):
    conf = st.session_state.get('pred_confidence', 0.0) if st.session_state.get('pred_confidence', 0.0) > 0.5 else 0.0
    placeholder.html(f"""
        <html>
        <head>
        <h1 style='text-align: center;'>Fault Detection</h1>
        <style>
            body {{
                font-family: sans-serif;
            }}
            .container {{
                max-width: 400px;
                margin: auto;
            }}
            .box {{
                background-color: white;
                border-radius: 1.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 0.5rem 0.5rem 0.5rem 0.5rem; 
                margin-bottom: 0.5rem;
                transition: box-shadow 0.3s ease;
                text-align: center;
            }}
            .box:hover {{
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }}
            .text {{
                color: #1f2937;
                font-size: 1.125rem;
                font-weight: 500;
                margin: 0;
            }}
            .highlight {{
                background-color: #fef3c7;
                border: 2px solid #f59e0b;
            }}
        </style>
        </head>
        <body>
        <div class="container">
            <div class="box {'highlight' if 'highlighted_state' in st.session_state and st.session_state['highlighted_state'] == 0 and st.session_state.get('pred_confidence', 0.0) > 0.5 else ''}"><p class="text">Normal state</p></div>
            <div class="box {'highlight' if 'highlighted_state' in st.session_state and st.session_state['highlighted_state'] == 1 and st.session_state.get('pred_confidence', 0.0) > 0.5 else ''}"><p class="text">Slow rate</p></div>
            <div class="box {'highlight' if 'highlighted_state' in st.session_state and st.session_state['highlighted_state'] == 2 and st.session_state.get('pred_confidence', 0.0) > 0.5 else ''}"><p class="text">Fast rate</p></div>
            <div class="box {'highlight' if 'highlighted_state' in st.session_state and st.session_state['highlighted_state'] == 3 and st.session_state.get('pred_confidence', 0.0) > 0.5 else ''}"><p class="text">Stop</p></div>
            <div class="box {'highlight' if 'highlighted_state' in st.session_state and st.session_state['highlighted_state'] == 4 and st.session_state.get('pred_confidence', 0.0) > 0.5 else ''}"><p class="text">Finger obstruction</p></div>
            <div class="box {'highlight' if 'highlighted_state' in st.session_state and st.session_state['highlighted_state'] == 5 and st.session_state.get('pred_confidence', 0.0) > 0.5 else ''}"><p class="text">Unbalanced pump</p></div>
            <div class="box {'highlight' if 'highlighted_state' in st.session_state and st.session_state['highlighted_state'] == 6 and st.session_state.get('pred_confidence', 0.0) > 0.5 else ''}"><p class="text">Occlusion</p></div>
            <div class="box {'highlight' if st.session_state.get('pred_confidence', 0.0) < 0.5 else ''}"><p class="text">Unknown</p></div>
        </div>
        <div class="container">
            <span style="font-size: 1.5rem; font-weight: 500; color: #1f2937;  padding-top: 20px;">Score: {conf:.3f}</span><br>
        </div>
        </body>
        </html>
    """)
if __name__ == "__main__":
    
    #if output/data.csv file exists, delete it
    try:
        import os
        if os.path.exists("output_data/data.csv"):
            os.remove("output_data/data.csv")
    except Exception as e:
        print(f"Error deleting existing data file: {e}")
    
    st.set_page_config(
    page_title="Motor fault detection",
    page_icon="🏂",
    layout="wide"
    )
    model_path = Path('model/best_model.keras')
    scaler_path = Path('model/scaler.pkl')
    #cache the model loading to avoid reloading it every time
    @st.cache_resource
    def load_model_cached():
        if model_path.exists() and scaler_path.exists():
            return load_model(model_path), joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model, scaler = load_model_cached()
    print("Model loaded successfully.")
    
    
    start_button = st.button("Start Data Acquisition")
    progress_bar = st.empty()
    download_button = st.empty()
    col = st.columns((2,2,1), gap='medium')
    fault_ui_placeholder = col[2].empty()
    with col[0]:
        st.markdown("<h3 style='text-align: center;'>X, Y, Z gee values</h3>", unsafe_allow_html=True)
        plot_placeholder = st.empty()
        
    with col[1]:
        st.markdown("<h3 style='text-align: center;'>Spectrum graph</h3>", unsafe_allow_html=True)
        fft_placeholder = st.empty()

    with col[2]:
        render_fault_ui(fault_ui_placeholder)
    if start_button:
        st.cache_data.clear()

        st.session_state['highlighted_state'] = ""
        st.session_state['pred_confidence'] = 0.0
        with col[2]:
            render_fault_ui(fault_ui_placeholder)
        #refresh all session_state variables
        if 'x_vals' in st.session_state:
            del st.session_state['x_vals']
            del st.session_state['y_vals']
            del st.session_state['z_vals']
        ser = wait_for_com_port(SERIAL_PORT, BAUD_RATE, timeout=0.01) #change COM number according to your PC
        ser.reset_input_buffer() 
        ser.reset_output_buffer()
        #reset done_event
        done_event.clear()
        # Start the background thread
        print("Starting background thread for serial data reading...")
        if serial_thread is None or not serial_thread.is_alive():
            serial_thread = threading.Thread(target=background_serial_read, daemon=True)
            serial_thread.start()
            t = time.time()
        else:
            print("Background thread is already running.")
        while not done_event.is_set():
            # st.toast("Reading sensor data...", icon="🔄")
            progress_bar.progress(progress_state["value"]/100.0, text=f"Reading sensor data... {progress_state['value']:.1f}%")
            if not messages_q.empty():
                message = messages_q.get()
                st.toast(message, icon="ℹ️")
            time.sleep(0.01)
        print(f'time taken to process data: {time.time() - t:.2f} seconds')
        
        if done_event.is_set():
            t = time.time()
            progress_bar.progress(progress_state["value"]/100, text="Finished reading sensor data. Generating CSV file...")
            
            x_vals = x_vals_q.get()
            y_vals = y_vals_q.get()
            z_vals = z_vals_q.get()

            x_vals = twos_complement_to_decimal_array(hex_strings_to_int_array(x_vals))
            y_vals = twos_complement_to_decimal_array(hex_strings_to_int_array(y_vals))
            z_vals = twos_complement_to_decimal_array(hex_strings_to_int_array(z_vals))

            try:
                df = pd.DataFrame({
                                'X': x_vals,
                                'Y': y_vals,
                                'Z': z_vals
                            })
                # save df to csv file called data.csv
                csv = df.to_csv("output_data/data.csv", index=False)
                csv = df.to_csv(index=False).encode('utf-8')
                download_button.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="data.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
            except Exception as e:
                download_button.error(f"Error creating CSV file: {e}")
            st.toast("CSV file created successfully! Plotting...", icon="✅")
            print(f'time taken to process data: {time.time() - t:.2f} seconds')
            with col[0]:
                try:
                    t = time.time()
                    fig = time_plot([x_vals, y_vals, z_vals], 0 , len(x_vals), "time_plot", n_noise_points=50) 
                    t = time.time()
                    plot_placeholder.pyplot(fig)
                    print(f'time taken to show plot on st: {time.time() - t:.2f} seconds')
                except Exception as e:
                    st.error(f"Error plotting time data: {e}")                    
            with col[1]:
                if not len(x_vals)==(SAMPLE_RATE*DURATION) or not len(y_vals)==(SAMPLE_RATE*DURATION) or not len(z_vals)==(SAMPLE_RATE*DURATION):
                    st.error(f"Sample length is not {SAMPLE_RATE*DURATION}. Please check the data acquisition process.")
                    st.error("x: " + str(len(x_vals)) + ", y: " + str(len(y_vals)) + ", z: " + str(len(z_vals)))
                else:
                    try:
                        t = time.time()
                        # Stack x,y,z time-series data into shape (64000, 3)
                        dataset = np.stack((x_vals, y_vals, z_vals), axis=-1)  # shape: (64000, 3)

                        # Add batch dimension to get shape (1, 64000, 3)
                        dataset = np.expand_dims(dataset, axis=0)

                        # Data Resampling
                        sensor_data_resampled = _downSampler(dataset[:, 50:, : ], 0, RESAMPLE_RATE)

                        #Noise removal
                        sensor_data= apply_wiener_filter(sensor_data_resampled)

                        #Scaling data
                        scaler = joblib.load(scaler_path)
                        sensor_data = _dataScaler(sensor_data, scaler)

                        fft_fig, len_xf, max_xf, fft_data = plot_fft_data(dataset, n_noise_points=50, scaler=scaler)
                        fft_placeholder.pyplot(fft_fig)
                        print(f'time taken to plot FFT data: {time.time() - t:.2f} seconds')
                        st.session_state['fft_fig'] = fft_fig
                        st.session_state['len_xf'] = len_xf
                        st.session_state['max_xf'] = max_xf
                    except Exception as e:
                        st.error(f"Error plotting FFT data: {e}")
            with col[2]:
                if len(fft_data) > 0:
                    try:
                        # Predict using the ML model
                        predictions = model.predict(sensor_data)
                        print(f"Predictions: {predictions}")
                        #there is only one sample in fft_data so get the first prediction
                        predicted_classes = np.argmax(predictions, axis=1)[0]
                        pred_confidence = round(np.max(predictions, axis=-1)[0],3)
                        
                        st.session_state['highlighted_state'] = predicted_classes
                        st.session_state['pred_confidence'] = pred_confidence
                        render_fault_ui(fault_ui_placeholder)
                    except Exception as e:
                        st.error(f"Error predicting faults: {e}")
    try:
        df = load_data()
        if not df.empty:

            x_vals = df['X'].tolist()
            y_vals = df['Y'].tolist()
            z_vals = df['Z'].tolist()

            # Sliders for X axis range (time index)
            with col[0]:
                x_start, x_end = st.slider(
                    "Select X-axis range",
                    0, int(len(x_vals)/SAMPLE_RATE), (0, int(len(x_vals)/SAMPLE_RATE)), step=1000
                )

                # Sliders for Y axis range (value range)
                # y_min_plot_1 = min(x_vals)
                # y_min_plot_2 = min(y_vals)
                # y_min_plot_3 = min(z_vals)
                # y_max_plot_1 = max(x_vals)
                # y_max_plot_2 = max(y_vals)
                # y_max_plot_3 = max(z_vals)
                # y_range_plot_1 = st.slider(
                #     "Select Y-axis range for acceleration X-axis plot",
                #     float(y_min_plot_1), float(y_max_plot_1), (float(y_min_plot_1), float(y_max_plot_1))
                # )
                # y_range_plot_2 = st.slider(
                #     "Select Y-axis range for acceleration Y-axis plot",
                #     float(y_min_plot_2), float(y_max_plot_2), (float(y_min_plot_2), float(y_max_plot_2))
                # )
                # y_range_plot_3 = st.slider(
                #     "Select Y-axis range for acceleration Z-axis plot",
                #     float(y_min_plot_3), float(y_max_plot_3), (float(y_min_plot_3), float(y_max_plot_3))
                # )

                # fig = time_plot([x_vals, y_vals, z_vals], x_start, x_end, "time_plot", y_range=[y_range_plot_1, y_range_plot_2, y_range_plot_3], n_noise_points=50) 
                fig = time_plot([x_vals, y_vals, z_vals], x_start, x_end, "time_plot", n_noise_points=50)
                plot_placeholder.pyplot(fig)
            if len(x_vals) == SAMPLE_RATE*DURATION and len(y_vals) == SAMPLE_RATE*DURATION and len(z_vals) == SAMPLE_RATE*DURATION:
                with col[1]:
                    x_start_fft, x_end_fft = st.slider(
                        "Select X-axis range",
                        0, int(st.session_state['max_xf']), (0, int(st.session_state['max_xf'])), step=500
                    )
                    # Stack x,y,z time-series data into shape (32000, 3)
                    dataset = np.stack((x_vals, y_vals, z_vals), axis=-1)  # shape: (32000, 3)

                    # Add batch dimension to get shape (1, 32000, 3)
                    dataset = np.expand_dims(dataset, axis=0)
                    fft_fig, len_xf , max_xf, fft_data= plot_fft_data(dataset, start=int(x_start_fft/st.session_state['max_xf']*st.session_state['len_xf']), stop=int(x_end_fft/st.session_state['max_xf']*st.session_state['len_xf']), n_noise_points=50, scaler=scaler)
                    fft_placeholder.pyplot(fft_fig)
    except FileNotFoundError:
        download_button.error("Data file not found. Please start data acquisition first.")
    