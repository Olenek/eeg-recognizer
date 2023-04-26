import numpy as np
from src.eeg_3d import gen_image, azim_proj


def psd_one_channel(channel_data, freq_bands, sample_rate):
    window_size = channel_data.shape[0]
    freq_bins = np.fft.fftfreq(window_size, d=1 / sample_rate)
    pos_freq_bins = freq_bins[:window_size // 2]
    fft_mag = np.abs(np.fft.fft(channel_data))[:window_size // 2] / (window_size / 2)

    # Calculate the PSD within each frequency band
    psd_channel = []
    for band, (low_freq, high_freq) in freq_bands.items():
        freq_mask = np.logical_and(pos_freq_bins >= low_freq, pos_freq_bins <= high_freq)
        psd_band = np.mean(fft_mag[freq_mask])
        psd_channel.append(psd_band)
    return psd_channel


def psd_from_eeg(eeg_data, selected_channels, freq_bands, window_size, step_size, sample_rate, two_dim=False):
    # Calculate the PSD within each frequency band for each window of data
    psd_data = []
    num_windows = (eeg_data.shape[0] - window_size) // step_size + 1
    for i in range(num_windows):
        window_start = i * step_size
        window_end = window_start + window_size

        # Apply FFT to each selected channel
        psd_window = []
        for channel in selected_channels:
            channel_data = eeg_data[window_start:window_end, channel]
            psd_channel = psd_one_channel(channel_data, freq_bands, sample_rate)
            if two_dim:
                psd_window.append(psd_channel)
            else:
                psd_window += psd_channel

        psd_data.append(psd_window)

    # Convert the PSD data to a numpy array
    return np.array(psd_data)


def images_from_eeg(eeg_data, selected_channels, loc_dict, freq_bands, window_size, step_size, sample_rate, resolution):
    num_windows = (eeg_data.shape[0] - window_size) // step_size + 1
    locs = np.array([azim_proj(loc_dict[i+1]) for i in selected_channels])
    images = []
    for i in range(num_windows):
        window_start = i * step_size
        window_end = window_start + window_size

        # Apply FFT to each selected channel
        psd_window = []
        for channel in selected_channels:
            channel_data = eeg_data[window_start:window_end, channel]
            psd_channel = psd_one_channel(channel_data, freq_bands, sample_rate)

            psd_window.append(psd_channel)
        psd_window = np.array(psd_window).T.flatten()  # window in format band1_channel1, band1_channel2, ...
        images.append(gen_image(locs=locs, features=psd_window, resolution=resolution))

    return np.array(images)
