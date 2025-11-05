'''
this file is used to make prediction of observations y_{1:T}

---  updated version of Evaluation.py

---  Predict the observation while plot the result (if the observation is 1D)
'''

import torch
import pandas as pd
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
from gpytorch.distributions import MultivariateNormal
from .GPModels import predictive_distribution
import sys
sys.path.append('../')
from src import utils as cg
cg.reset_seed(0)

def plotPred(y_pred, lower, upper, groundTruth, ifPlot, ifSave, savePath):
    """
    :param y_pred:          shape: seq_len x batch_size x output_dim
    :param lower:           shape: seq_len x batch_size x output_dim
    :param upper:           shape: seq_len x batch_size x output_dim
    :param groundTruth:     shape: batch_size x seq_len x output_dim
    :param ifPlot:
    :param ifSave:
    :param savePath:
    :return:
    """
    batch_size= groundTruth.shape[0]
    seq_len = groundTruth.shape[1]
    output_dim = groundTruth.shape[-1]

    y_pred = y_pred.transpose(1, 0)  # shape: batch_size x seq_len x output_dim
    lower = lower.transpose(1, 0)   # shape: batch_size x seq_len x output_dim
    upper = upper.transpose(1, 0)   # shape: batch_size x seq_len x output_dim


    '''  
    将所有的batch 的数据合起来成一整条长序列:  # shape: [batch_size x seq_len] x output_dim
    '''
    y_pred_oneSeq = y_pred.reshape(batch_size*seq_len, output_dim)
    lower_oneSeq = lower.reshape(batch_size*seq_len, output_dim)
    upper_oneSeq = upper.reshape(batch_size * seq_len, output_dim)
    groundTruth_oneSeq = groundTruth.reshape(batch_size * seq_len, output_dim)

    '''  
    currently let us assume the output_dim = 1
    '''
    assert (output_dim==1)
    f, ax = plt.subplots(1, 1, figsize=(15, 10))

    fontsize = 28

    T = np.arange(1, batch_size*seq_len+1)

    # Plot test data as black starss
    ax.plot(T, groundTruth_oneSeq.cpu().numpy().reshape(-1, ), 'r', label='observation', markersize=10)

    # Plot test data as read stars
    ax.plot(T, y_pred_oneSeq.detach().cpu().numpy(), 'b', label='prediction', markersize=10 )

    # Shade between the lower and upper confidence bounds
    ax.fill_between(T, lower_oneSeq.squeeze().detach().cpu().numpy(),
                    upper_oneSeq.squeeze().detach().cpu().numpy(), alpha=0.5, label='95% CI')

    ax.legend(loc='upper left', fontsize=fontsize)
    # plt.title(f"Epoch: {epoch}", fontsize=15)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if ifSave:
        plt.savefig(savePath + "seriesPrediction.pdf")
    if ifPlot:
        plt.show()


def predictive_observations(model, output_test, x_0=None, input_test=None, emi_idx=None, testSet=None,
                            ifPlot=True, ifSave=True, savePath=None):
    """
    test transition function GP
    Args:
        model:          GPSSMs_all
        output_test:    observations:   shape: batch_size x seq_len x output_dim
        input_test:     control inputs: shape: batch_size x seq_len x input_dim
        x_0:            initial state: shape:  batch_size x state_dim
        emi_idx:        emission index [i]
        testSet:
        ifPlot:         flag to show if plot the figure
        ifSave:         flag to show if save the figure
        savePath:       Path to save the plots
    """
    model.eval()

    device = output_test.device
    batch_size = output_test.shape[0]
    pred_len = output_test.shape[1]

    # emission index: indicates which dimension of latent state is observable
    if emi_idx is None:
        emi_idx = [0]
    assert (len(emi_idx) == model.output_dim)

    # emission noise variance: batch_size x output_dim x output_dim
    obser_covar = torch.diag_embed(model.emission_likelihood.noise.view(-1)).repeat(batch_size, 1, 1)


    ''' 
    ---------------------  ----------------------------------------------------------------------------------
    1.  using recognition network to get latent state 
    can also obtain the first prediction latent state by using the last state got from observable sequence
    ---------------------------------------------------------------------------------------------------------
     '''
    if x_0 is None:
        q_x_0 = model.recognet(output_test, input_test)   # shape: batch_size x dim_state
        x_0 = q_x_0.rsample()                             # shape: batch_size x dim_state

    # initialization:
    pred_likelihood = torch.tensor(0.).to(device)
    y_pred_all = []
    y_pred_var_all = []
    x_t_1 = x_0

    for t in range(pred_len):
        gp_input = x_t_1                                  # shape:  batch_size x dim_state
        if input_test is not None:
            c_t = input_test[:, t]                        # shape: batch_size x dim_input
            gp_input = torch.cat((c_t, x_t_1), dim=-1)    # shape: batch_size x (dim_input + dim_state)
        gp_input = gp_input.repeat(model.state_dim, 1, 1) # shape: dim_state x batch_size x (dim_input + dim_state)

        # m1: dim_state x batch_size;  m2: dim_state x batch_size
        m1, m2, qf_mean, qf_variance = predictive_distribution(model.transition, model.likelihood, model.flow, gp_input)

        x_t = m1.transpose(0, 1)      # shape: batch_size x state_dim
        x_t_var = m2.transpose(0, 1)  # shape: batch_size x state_dim

        # emission model: the mean
        yt_mean = x_t[:, emi_idx]       # shape: batch_size x output_dim
        yt_var_1 = x_t_var[:, emi_idx]  # shape: batch_size x output_dim

        # emission model: the variance
        output_covar = obser_covar + torch.diag_embed( yt_var_1 ) # batch_size x output_dim x output_dim

        # construct the output distribution (approximate Gaussian)
        pyt = MultivariateNormal(yt_mean, output_covar)  # shape: batch_size x output_dim

        # read the test dataset
        y_tmp = output_test[:, t, :]                     # shape: batch_size x output_dim

        # compute the prediction likelihood
        pred_likelihood = pred_likelihood + pyt.log_prob(y_tmp).mean().div(pred_len)  # average batch

        # update x_t_1
        x_t_1 = x_t

        # save prediction
        y_pred_all.append(yt_mean)
        y_pred_var_all.append( yt_var_1 + model.emission_likelihood.noise.view(-1).repeat(batch_size, 1) )

        """ -------------------------------------------- loop end -----------------------------------------------"""

    """ --------------------------------------------------------------------------------------------------------"""
    y_pred = torch.stack(y_pred_all, dim=0)              # shape: seq_len x batch_size x output_dim
    y_pred_var = torch.stack(y_pred_var_all, dim=0)      # shape: seq_len x batch_size x output_dim
    y_pred_std = y_pred_var.sqrt()                       # shape: seq_len x batch_size x output_dim

    lower, upper = y_pred - 2 * y_pred_std, y_pred + 2 * y_pred_std   # shape: seq_len x batch_size x output_dim



    """ --------------------------------------------------------------------------------------------------------"""
    if testSet is not None:
        mean_dataset = torch.tensor(testSet.output_normalizer.mean, device=output_test.device)
        std_dataset = torch.tensor(testSet.output_normalizer.sd, device=output_test.device)

        output_test = output_test * std_dataset + mean_dataset
        y_pred = y_pred * std_dataset + mean_dataset
        lower = lower * std_dataset + mean_dataset
        upper = upper * std_dataset + mean_dataset
    """ --------------------------------------------------------------------------------------------------------"""


    MSE_loss = torch.nn.MSELoss(reduction='none')         # element-wise squared loss
    MSE = MSE_loss( y_pred.transpose(1,0), output_test )  # shape: batch_size x seq_len x output_dim
    MSE_dim = MSE.sum(dim=-1)                             # shape: batch_size x seq_len
    MSE_ave = MSE_dim.mean(dim=[0,1])
    RMSE_ave = MSE_ave.sqrt()

    # print(f"MSE: {MSE_dim}")
    if ifPlot or ifSave:
        plotPred(y_pred, lower, upper, output_test, ifPlot, ifSave, savePath)


    return pred_likelihood.item(), MSE, MSE_ave.item(), RMSE_ave.item()

