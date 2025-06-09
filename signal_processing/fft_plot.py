from .utils import  fft_plot, _downSampler, _FFT, _dataScaler, RESAMPLE_RATE
import numpy as np
def plot_fft_data(x_vals, y_vals, z_vals, start=None, stop=None):
    # Data Resampling
    data_x_resampled = _downSampler([np.array(x_vals).reshape(-1, 1)], 0, RESAMPLE_RATE)
    # print("Data_x_resampled", data_x_resampled.shape)
    data_x_resampled_fft = _FFT(data_x_resampled)
    # print("Data_x_resampled_fft", data_x_resampled_fft.shape)
    data_y_resampled = _downSampler([np.array(y_vals).reshape(-1, 1)], 0, RESAMPLE_RATE)
    data_y_resampled_fft = _FFT(data_y_resampled)
    data_z_resampled = _downSampler([np.array(z_vals).reshape(-1, 1)], 0, RESAMPLE_RATE)
    data_z_resampled_fft = _FFT(data_z_resampled)
    fig = fft_plot([data_x_resampled_fft, data_y_resampled_fft, data_z_resampled_fft], start, stop, 'fft_normal')
    return fig

