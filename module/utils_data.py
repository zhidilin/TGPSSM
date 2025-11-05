'''
 get datasets and normalize it
'''

import torch
from torch.utils import data
import numpy as np
import scipy.io as sio
import os


def normalizer(data_in_train, data_out_train):
    """
    data_in_train: np.array, shape: N x dim_input
    data_out_train: np.array, shape: N x dim_output
    """
    U_mean = data_in_train.mean(axis=0)
    Y_mean = data_out_train.mean(axis=0)

    U_std = data_in_train.std(axis=0)
    Y_std = data_out_train.std(axis=0)

    # normalized and shifted original data, such that the shifted data has zero-mean and unit variance
    U_normalized = (data_in_train - U_mean)/U_std
    Y_normalized = (data_out_train - Y_mean)/Y_std

    return U_normalized, Y_normalized, U_mean, U_std, Y_mean, Y_std

def get_realdata(DIR, data_name = 'actuator'):
    if data_name == 'actuator':
        data_name = 'actuator.mat'
        #
        # Datasets can be downloaded from:
        # [1] http://www.iau.dtu.dk/nnbook/systems.html
        # [2] https://github.com/zhenwendai/RGP/tree/master/datasets/system_identification
        #
        split_point = 512
        data = sio.loadmat(os.path.join(DIR, data_name))
        data_in = data['u'].astype(np.float64)
        data_out = data['p'].astype(np.float64)

    elif data_name == 'ballbeam':
        data_name = 'ballbeam.dat'
        #
        # Datasets from http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
        #
        data = np.loadtxt(os.path.join(DIR, data_name))
        split_point = 500
        data_in = data[:, 0]
        data_out = data[:, 1]

    elif data_name == 'drive':
        data_name = 'drive.mat'
        #
        # Datasets from http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
        #
        data = sio.loadmat(os.path.join(DIR, data_name))
        split_point = 250
        data_in = data['u1']
        data_out = data['z1']

    elif data_name == 'gas_furnace':
        data_name = 'gas_furnace.csv'
        #
        # Datasets from http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
        #
        data = np.loadtxt(os.path.join(DIR, data_name), skiprows=1, delimiter=',')

        split_point = 148
        data_in = data[:, 0]
        data_out = data[:, 1]

    elif data_name == 'dryer':
        data_name = 'dryer.dat'
        #
        # Datasets from http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
        #
        data = np.loadtxt(os.path.join(DIR, data_name))
        split_point = 500
        data_in = data[:, 0]
        data_out = data[:, 1]
    else:
        raise NotImplementedError("Dataset does not exist")

    data_in_train = data_in[:split_point]            # U_train
    data_out_train = data_out[:split_point]          # Y_train

    data_in_test = data_in[split_point:]             # U_test
    data_out_test = data_out[split_point:]           # Y_test

    U_normalized_train, Y_normalized_train, U_mean, U_std, Y_mean, Y_std = normalizer(data_in_train, data_out_train)
    # normalize the test data based upon the [mean, std] pairs from training data
    U_normalized_test = (data_in_test - U_mean) / U_std
    Y_normalized_test = (data_out_test - Y_mean) / Y_std

    return U_normalized_train.reshape(-1, 1), Y_normalized_train.reshape(-1, 1),\
           U_normalized_test.reshape(-1, 1), Y_normalized_test.reshape(-1, 1)

def get_minibatch(obsv, batch_size):
    """
    get minibatch data
    Input:  Tensor
        obsv: observed time series. Shape: [episodes, seq_len, obser_dim]
        batch_size: batch size
    Output: List of Tensor
        shape: [ batch_size, seq_len, obser_dim]
    """
    N_traj = obsv.shape[0]                  # number of trajectories
    indices = torch.randperm(N_traj)        # make the order of index random

    batch_data_list = []
    num_batch = int(N_traj / batch_size)    # number of batch in the full dataset
    for i in range(num_batch):
        index = indices[i * batch_size: (i + 1) * batch_size]   # i-th batch index
        batch_data = obsv[index,]                               # i-th batch data
        batch_data_list.append(batch_data)
    if N_traj % batch_size:
        num_batch = int(N_traj / batch_size) + 1             # number of batch in the full dataset
        for i in range(num_batch-1):
            index = indices[i*batch_size: (i+1)*batch_size]  # i-th batch index
            batch_data = obsv[index,]                        # i-th batch data
            batch_data_list.append(batch_data)

        index = indices[N_traj-batch_size: N_traj]           # last batch index
        batch_data = obsv[index,]
        batch_data_list.append(batch_data)

    else:
        num_batch = int(N_traj/batch_size)                   # number of batch in the full dataset
        for i in range(num_batch):
            index = indices[i*batch_size: (i+1)*batch_size]  # i-th batch index
            batch_data = obsv[index,]                         # i-th batch data
            batch_data_list.append(batch_data)

    return num_batch, batch_data_list

def traj_2_batch(U_full, seq_len, window_slides):
    """
    make the long sequence data into several sub-trajectories
    Input: numpy data
        U_full:         Shape: T x input_dim

        seq_len:        the length of sub-trajectories

        window_slides:  index interval between two sub-trajectories,  should be less than seq_len
                        e.g, the first trajectory with first element x[0],
                             the next trajectory with first element x[0+window_slides],
    Output: Tensor
        U_batch:  Shape: episodes x seq_len x input_dim
    """
    assert window_slides<=seq_len,  'window_slides shoudl be less than seq_len'
    assert len(U_full.shape) == 2,  'The dimension should be T x input_dim'
    T  = U_full.shape[0]            # length of long trajectory
    dim  = U_full.shape[-1]         # dimension of the data

    # how many episodes of sub-trajectories, where the length of sub-trajectories = seq_len
    episodes = int( (T - seq_len)/window_slides ) + 1      # according to  T = seq_len + window_slides * (episodes-1)

    if (T - seq_len) % window_slides:
        episodes = episodes + 1

    # initialize U_tensor
    U_batch = torch.zeros(episodes, seq_len, dim)
    if (T - seq_len) % window_slides:
        # 序列长度不是batch_size的整数倍，则最后一条子序列复用倒数第二条序列数据
        # If the sequence length is not an integer multiple of batch_size,
        # the last subsequence reuses the penultimate sequence data
        for j in range(episodes-1):
            U_batch[j] = U_full[ j * window_slides : j * window_slides + seq_len ]
        U_batch[j+1] = U_full[T-seq_len : T]

    else:
        for j in range(episodes):
            U_batch[j] =  U_full[ j * window_slides : j * window_slides + seq_len ]

    return U_batch, episodes
