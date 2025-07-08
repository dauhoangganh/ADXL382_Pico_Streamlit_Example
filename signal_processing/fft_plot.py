from .utils import  fft_plot, _downSampler, _FFT, _dataScaler, RESAMPLE_RATE
import numpy as np
def plot_fft_data(dataset, start=None, stop=None, n_noise_points=0, scaler=None):  
    # Data Resampling
    dataset_resampled = _downSampler(dataset[:, n_noise_points:, : ], 0, RESAMPLE_RATE)
    #FFT Transformation
    dataset_resampled_fft = _FFT(dataset_resampled)
    # Data Scaling
    dataset_resampled_fft_scaled = _dataScaler(dataset_resampled_fft, scaler)
    # Plotting
    fig, lenxf , max_xf = fft_plot(dataset_resampled_fft_scaled, start, stop, 'fft_normal')
    return fig, lenxf, max_xf, dataset_resampled_fft_scaled
