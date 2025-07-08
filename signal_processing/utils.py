"""
Any utility function that is required for data exploratory analysis goes here
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfftfreq, rfft
from numba import njit
from sklearn.preprocessing import StandardScaler
import scipy.signal
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

def time_plot(yf: list, start: int, stop: int, fname: str, y_range=None, n_noise_points=0):
    '''
    Plots a time series

    params:
    ---
    yf (list): list of input data to plot
    start (int): start time (s)
    stop (int): stop time (s)
    fname (str): save file name
    y_range (list): list of y-axis limits for each axis, e.g. [[-2, 2], [-2, 2], [-2, 2]]
    n_noise_points (int): number of noise points to remove from the plot (default is 0, no noise)
    '''
    #remove first n_noise_point from each axis
    if n_noise_points > 0:
        yf = [axis[n_noise_points:] for axis in yf]
    fig, ax = plt.subplots(3, 1, tight_layout=True, figsize=(10, 8), sharex=True)
    plt.rcParams['font.size'] = '16'
    start = int(start * SAMPLE_RATE)
    stop = int(stop * SAMPLE_RATE) if stop is not None else len(yf[0])  # Ensure stop is within bounds
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
    yf (list): list of input data to plot, shape=(n_sample=1, n_freq_bins, n_features=3)
    fname (str): save file name
    '''
    # N = int((SAMPLE_RATE / RESAMPLE_RATE) * DURATION)
    # xf = rfftfreq(N-1, 1 / int(SAMPLE_RATE / RESAMPLE_RATE))
    fig, ax = plt.subplots(3, 1, tight_layout=True, figsize=(10, 8), sharex=True)
    plt.rcParams['font.size'] = '16'
    if start is None:
        start=0
    
    if yf[0].shape[0] * 2 !=  SAMPLE_RATE*DURATION or yf[0].shape[0] * 2 != SAMPLE_RATE*DURATION or yf[0].shape[0] * 2 != SAMPLE_RATE*DURATION:
        N = [(yf[0].shape[0]) * 2 , (yf[0].shape[0]) * 2, (yf[0].shape[0]) * 2] # reverse engineer from rfft length
        xf = [rfftfreq(N[0] - 1, 1 / int(SAMPLE_RATE / RESAMPLE_RATE)) , rfftfreq(N[1] - 1, 1 / int(SAMPLE_RATE / RESAMPLE_RATE)), rfftfreq(N[2] - 1, 1 / int(SAMPLE_RATE / RESAMPLE_RATE))]
        if stop is None:
            stop = max(len(xf[0]), len(xf[1]), len(xf[2]))
        ax[0].plot(xf[0][start:stop], yf[0][start:stop, [0]], label='X-axis', color='r')
        ax[1].plot(xf[1][start:stop], yf[0][start:stop, [1]], label='Y-axis', color='g')
        ax[2].plot(xf[2][start:stop], yf[0][start:stop, [2]], label='Z-axis', color='b')
        lenxf = xf[0].shape[0]  # Assuming all xf have the same length
        max_xf = int(max(xf[0]))
    else:
        N = int((SAMPLE_RATE / RESAMPLE_RATE) * DURATION)
        xf = rfftfreq(N-1, 1 / int(SAMPLE_RATE / RESAMPLE_RATE))
        lenxf = len(xf)
        max_xf = int(max(xf))
        if stop is None:
            stop = len(xf)
        ax[0].plot(xf[start:stop], yf[0][start:stop, [0]], label='X-axis', color='r')
        ax[1].plot(xf[start:stop], yf[0][start:stop, [1]], label='Y-axis', color='g')
        ax[2].plot(xf[start:stop], yf[0][start:stop, [2]], label='Z-axis', color='b')
    
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
    return fig, lenxf, max_xf

def apply_wiener_filter(data, mysize=31):
    """
    Apply Wiener filter to 3D time-series data (samples, time steps, features).

    Parameters:
        data (np.ndarray): Input shape (n_samples, n_steps, n_features)
        mysize (int): Size of Wiener filter window

    Returns:
        np.ndarray: Denoised data with same shape
    """
    denoised = np.empty_like(data)

    for i in range(data.shape[0]):           # loop over samples
        for j in range(data.shape[2]):       # loop over features (e.g., x/y/z)
            denoised[i, :, j] = scipy.signal.wiener(data[i, :, j], mysize=mysize, noise=1e-8)


    return denoised

def get_scaler(X_train):
    X_train_2d = X_train.reshape(-1, X_train.shape[2])#reshapes 3D array into a 2D array where all the time steps from all samples are stacked together while still having 3 feature columns.
    scaler = StandardScaler() #learn the mean and std of train_data per feature, across all frequency bins
    scaler.fit(X_train_2d)
    return scaler

def _dataScaler(data: list, scaler) -> list:
    '''
    Reads in data and returns a scaled list.

    params:
    ---
    data (list): a list of data to scale--->(3D array (samples, time steps, features))

    returns:
    ---
    final_sequence (list): resampled data
    '''


    # Reshape for scaling
    data_2d = data.reshape(-1, data.shape[2])#reshapes 3D array into a 2D array where all the time steps from all samples are stacked together while still having 3 feature columns.
    data_scaled = scaler.transform(data_2d)
    data_scaled = data_scaled.reshape(data.shape)
    return data_scaled


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
        data_fft.append(np.stack(np.abs(rfft(dataset, axis=0))[1:, :]))
    # print("Data FFT shape:", np.array(data_fft).shape)
    return np.stack(data_fft)


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