from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np
import pandas as pd
import pickle
from scipy.linalg import sqrtm
from TSGBench.src.ts2vec import initialize_ts2vec
import torch

from library.constants import WINDOW_SIZE

with open("dashboard/fid_model.pkl", 'rb') as f:
    fid_model = pickle.load(f)

# fid_model = initialize_ts2vec(np.transpose(ori_data, (0, 2, 1)),torch.device('cpu'))

def find_length(data):
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    base = 3
    nobs = len(data)
    nlags = int(min(10 * np.log10(nobs), nobs - 1))
    auto_corr = acf(data, nlags=nlags, fft=True)[base:]
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max] < 3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except Exception as e:
        print(e)
        return 125

def preprocess(data, step=1):
    window_all = []
    for i in range(data.shape[1]):
        window_all.append(find_length(data[:,i]))

    seq_length = int(np.mean(np.array(window_all)))

    window_size = 125#seq_length
    
    if data.ndim != 2:
        raise ValueError("Input array must be 2D")
    L, C = data.shape  # Length and Channels
    if L < window_size:
        raise ValueError("Window size must be less than or equal to the length of the array")

    # Calculate the number of windows B
    B = L - window_size + 1
    
    # Shape of the output array
    new_shape = (B, window_size, C)
    
    # Calculate strides
    original_strides = data.strides
    new_strides = (original_strides[0],) + original_strides  # (stride for L, stride for W, stride for C)

    # Create the sliding window view
    strided_array = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)
    return strided_array


    
def calculate_fid(ori_data, gen_data, fid_model=fid_model):
    ori_data = ori_data.numpy()
    gen_data = gen_data.numpy()
    ori_data = preprocess(ori_data)
    gen_data = preprocess(gen_data)
    # fid_model = initialize_ts2vec(np.transpose(ori_data, (0, 2, 1)),torch.device('cpu'))

    ori_repr = fid_model.encode(np.transpose(ori_data,(0, 2, 1)), encoding_window='full_series')
    gen_repr = fid_model.encode(np.transpose(gen_data,(0, 2, 1)), encoding_window='full_series')
    # calculate mean and covariance statistics
    mu1, sigma1 = ori_repr.mean(axis=0), np.cov(ori_repr, rowvar=False)
    mu2, sigma2 = gen_repr.mean(axis=0), np.cov(gen_repr, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def evaluate_data(ori_data, gen_data):
    train_data = ori_data
    fid_model = initialize_ts2vec(np.transpose(train_data, (0, 2, 1)),torch.device('cpu'))
    ori_repr = fid_model.encode(np.transpose(ori_data,(0, 2, 1)), encoding_window='full_series')
    gen_repr = fid_model.encode(np.transpose(gen_data,(0, 2, 1)), encoding_window='full_series')
    cfid = calculate_fid(ori_repr,gen_repr)
    return cfid