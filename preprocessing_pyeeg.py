import os
import pickle

import numpy as np
import pyeeg as pe
from tqdm import tqdm


def fft_processing(filename, channel, band, window_size, step_size, sample_rate):
    '''
    arguments:  string subject
                list channel indice
                list band
                int window size for FFT
                int step size for FFT
                int sample rate for FFT
    return:     void
    '''
    labels_out = []
    features_out = []
    out_name = os.path.split(filename)[-1].split('.')[0]  # a/b/c.py -> c
    labels_dir = os.path.join('data_processed', 'labels')
    features_dir = os.path.join('data_processed', 'features')

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    with open(filename, 'rb') as f:
        subject_data = pickle.load(f, encoding='latin1')  # resolve the python 2 data problem by encoding : latin1
        for i in range(0, 40):
            # loop over 0-39 trails
            data = subject_data["data"][i]
            labels = subject_data["labels"][i]
            start = 0

            while start + window_size < data.shape[1]:
                meta_data = []  # meta vector for analysis
                for j in channel:
                    # FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
                    y = pe.bin_power(data[j][start: start + window_size], band,
                                     sample_rate)
                    meta_data = meta_data + list(y[0])

                features_out.append(np.array(meta_data))
                labels_out.append(np.array(labels))

                start = start + step_size

        np.save(os.path.join(features_dir, out_name), np.array(features_out), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(labels_dir, out_name), np.array(labels_out), allow_pickle=True, fix_imports=True)


channels = [1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31]  # 14 Channels chosen to fit Emotiv Epoch+
bands = [4, 8, 12, 16, 25, 45]  # 5 bands
window_size = 256  # Averaging band power of 2 sec
step_size = 16  # Each 0.125 sec update once
sample_rate = 128  # Sampling rate of 128 Hz
#
#     ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']
# #List of subjects

data_directory = os.path.join('data', 'data_preprocessed_python')

for file in tqdm(os.listdir(data_directory)):
    fullpath = os.path.join(data_directory, file)
    fft_processing(fullpath, channels, bands, window_size, step_size, sample_rate)
