"""
Any utility function that is required for data exploratory analysis goes here
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfftfreq, rfft
from sklearn.preprocessing import MinMaxScaler
from numba import njit
DURATION = 4 # Time-series duration in seconds
OUTPUT_DATA_DIR = Path('output_data')  # Directory to save output plots
SAMPLE_RATE = 16000  # Original sample rate in Hz
RESAMPLE_RATE = 1  # Resample rate used to desample the time-series
def ensure_output_dir_exists():
    '''
    Ensures that the output directory exists.
    Creates it if it does not exist.
    '''
    if not OUTPUT_DATA_DIR.exists():
        OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

def time_plot(yf: list, start: int, stop: int, fname: str, y_range=None):
    '''
    Plots a time series

    params:
    ---
    yf (list): list of input data to plot
    start (int): start time (s)
    stop (int): stop time (s)
    fname (str): save file name
    '''
    # time = np.linspace(0, DURATION, len(yf), endpoint=False)

    fig, ax = plt.subplots(3, 1, tight_layout=True, figsize=(10, 8), sharex=True)
    plt.rcParams['font.size'] = '16'

    ax[0].plot(np.linspace(0, len(yf[0])/SAMPLE_RATE, len(yf[0]), endpoint=False)[start:stop], yf[0][start:stop], label='X-axis', color='r')
    ax[1].plot(np.linspace(0, len(yf[1])/SAMPLE_RATE, len(yf[1]), endpoint=False)[start:stop], yf[1][start:stop], label='Y-axis', color='g')
    ax[2].plot(np.linspace(0, len(yf[2])/SAMPLE_RATE, len(yf[2]), endpoint=False)[start:stop], yf[2][start:stop], label='Z-axis', color='b')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    # for label in (ax[2].get_xticklabels() + ax[2].get_yticklabels()):
    #     label.set_fontsize(14)

    ax[2].set_xlabel("Time [s]", fontsize=16)
    if y_range:
        ax[0].set_ylim(y_range[0])
        ax[1].set_ylim(y_range[1])
        ax[2].set_ylim(y_range[2])
    # Common y-axis label for the whole figure
    fig.text(0.04, 0.5, 'g value', va='center', rotation='vertical')
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # leave space for common labels and title
    # ensure_output_dir_exists()
    # file_location = OUTPUT_DATA_DIR / Path(f'{fname}.png')
    # plt.savefig(file_location)
    return fig

def fft_plot(yf: list, start=None, stop=None, fname='fft_plot'):
    '''
    Plots the FFT

    params:
    ---
    yf (list): list of input data to plot
    fname (str): save file name
    '''
    # N = int((SAMPLE_RATE / RESAMPLE_RATE) * DURATION)
    # xf = rfftfreq(N-1, 1 / int(SAMPLE_RATE / RESAMPLE_RATE))
    fig, ax = plt.subplots(3, 1, tight_layout=True, figsize=(10, 8), sharex=True)
    plt.rcParams['font.size'] = '16'
    if start is None:
        start=0
    
    if yf[0][0].shape[0] * 2 != yf[1][0].shape[0] * 2 or yf[0][0].shape[0] * 2 != yf[2][0].shape[0] * 2:
        N = [(yf[0][0].shape[0]) * 2 , (yf[1][0].shape[0]) * 2, (yf[2][0].shape[0]) * 2] # reverse engineer from rfft length
        xf = [rfftfreq(N[0] - 1, 1 / int(SAMPLE_RATE / RESAMPLE_RATE)) , rfftfreq(N[1] - 1, 1 / int(SAMPLE_RATE / RESAMPLE_RATE)), rfftfreq(N[2] - 1, 1 / int(SAMPLE_RATE / RESAMPLE_RATE))]
        if stop is None:
            stop = max(len(xf[0]), len(xf[1]), len(xf[2]))
        ax[0].plot(xf[0][start:stop], yf[0][0][start:stop], label='X-axis', color='r')
        ax[1].plot(xf[1][start:stop], yf[1][0][start:stop], label='Y-axis', color='g')
        ax[2].plot(xf[2][start:stop], yf[2][0][start:stop], label='Z-axis', color='b')
    else:
        N = int((SAMPLE_RATE / RESAMPLE_RATE) * DURATION)
        xf = rfftfreq(N-1, 1 / int(SAMPLE_RATE / RESAMPLE_RATE))
        if stop is None:
            stop = len(xf)
        ax[0].plot(xf[start:stop], yf[0][0][start:stop], label='X-axis', color='r')
        ax[1].plot(xf[start:stop], yf[1][0][start:stop], label='Y-axis', color='g')
        ax[2].plot(xf[start:stop], yf[2][0][start:stop], label='Z-axis', color='b')
    
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    # for label in (ax[2].get_xticklabels() + ax[2].get_yticklabels()):
    #     label.set_fontsize(14)
    
    # Common x-axis label (at bottom subplot)
    ax[2].set_xlabel("Frequency (Hz)", fontsize=16)
    
    # Common y-axis label for the whole figure
    fig.text(0.04, 0.5, 'Signal strength', va='center', rotation='vertical')
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # leave space for common labels and title
    # ensure_output_dir_exists
    # file_location = OUTPUT_DATA_DIR / Path(f'{fname}.png')
    # plt.savefig(file_location)
    return fig, len(xf)

def _dataScaler(data: list) -> list:
    '''
    Reads in data and returns a scaled list.

    params:
    ---
    data (list): data to down sample

    returns:
    ---
    final_sequence (list): resampled data
    '''
    data_temp = np.reshape(data, (-1, data.shape[2]))
    norm = MinMaxScaler().fit(data_temp)
    data_norm = norm.transform(data_temp)
    data_final = np.reshape(data_norm, (-1, data.shape[1], data.shape[2]))

    return data_final


def _downSampler(data: list, start_index: int, sample_rate: int) -> list:
    '''
    Reads in raw data from .csv files and returns a resampled list

    params:
    ---
    data (list): data to down sample
    start_index (int): starting index
    sample_rate (int): sampling rate

    returns:
    ---
    final_sequence (list): resampled data
    '''
    final_sequence = list()
    for dataset in data:
        data_resampled = []
        start = start_index
        stop = sample_rate
        for i in range(int(len(dataset)/sample_rate)):
            data_resampled.append(dataset[start:stop, :].mean(axis=0))
            start += sample_rate
            stop += sample_rate
        final_sequence.append(np.stack(data_resampled))

    return np.stack(final_sequence)


def _FFT(data: list) -> list:
    '''
    Reads in resampled data and performs a Fast Fourier Transform with DC offset removal

    params:
    ---
    data: data to perform Fast Fourier Transform

    returns:
    ---
    data_fft (list): FFT data
    '''
    data_fft = list()
    for dataset in data:
        data_fft.append(np.abs(rfft(dataset, axis=0))[1:, :])
    # print("Data FFT shape:", np.array(data_fft).shape)
    return np.stack(data_fft)

# def twos_complement_to_decimal(hex_str):
#     value = int(hex_str, 16)
#     if value & 0x8000:  # if the sign bit is set (for 16-bit numbers)
#         value -= 0x10000
#     return value

@njit
def twos_complement_to_decimal_array(int_list):
    output = []
    for val in int_list:
        if val & 0x8000:
            val -= 0x10000
        output.append(val* 500e-6)  # Convert to g value by scaling with ADXL382 scale factor
    return output

def hex_strings_to_int_array(hex_bytes_list):
    # Decode hex strings and convert to int
    return np.array([int(h.decode(), 16) for h in hex_bytes_list], dtype=np.uint16)