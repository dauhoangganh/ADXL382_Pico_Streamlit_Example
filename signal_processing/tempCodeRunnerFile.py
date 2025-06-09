from utils import  fft_plot, _downSampler, _FFT, _dataScaler, RESAMPLE_RATE
import numpy as np
def plot_fft_data(x_vals, y_vals, z_vals):
    # Data Resampling
    data_x_resampled = _downSampler([np.array(x_vals).reshape(-1, 1)], 0, RESAMPLE_RATE)
    print("Data_x_resampled", data_x_resampled)
    data_x_resampled_fft = _FFT(data_x_resampled)
    print("Data_x_resampled_fft", data_x_resampled_fft)
    data_y_resampled = _downSampler([np.array(y_vals).reshape(-1, 1)], 0, RESAMPLE_RATE)
    data_y_resampled_fft = _FFT(data_y_resampled)
    data_z_resampled = _downSampler([np.array(z_vals).reshape(-1, 1)], 0, RESAMPLE_RATE)
    data_z_resampled_fft = _FFT(data_z_resampled)
    fig = fft_plot(np.stack([data_x_resampled, data_y_resampled, data_z_resampled]), 'fft_normal_resampled')
    return fig

plot_fft_data(np.random.randint(0, 101, size=100), np.random.randint(0, 101, size=100),np.random.randint(0, 101, size=100))