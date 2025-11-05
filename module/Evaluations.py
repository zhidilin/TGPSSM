'''
this file is used to make prediction of observations y_{1:T}
'''

import torch
import pandas as pd
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.distributions import MultivariateNormal
import sys
sys.path.append('../')
from src import utils as cg
cg.reset_seed(0)
device = cg.device

def predictive_observations(model, output_test, cov_g, data_sd, data_mean, epoch,
                            x_0=None,
                            input_test=None,
                            emi_idx=None,
                            N_MC=50,
                            plt_save=True,
                            result_dir='./results/prediction/'):
    """
    test transition function GP
    Args:
        model:          GPSSM, JOGPSSM, COGPSSM, JOTGPSSM_E, COTGPSSM_E, JOTGPSSM_nvp, COTGPSSM_nvp
        output_test:    observations:   shape: batch_size x seq_len x output_dim
        input_test:     control inputs: shape: batch_size x seq_len x input_dim
        cov_g:          learned observation noise
        x_0:            initial state: shape:  batch_size x state_dim
        emi_idx:        emission index [i]
        pred_len:       prediction length: model.seq_len
        N_MC:           trajectory number (particle number)
    """
    dtype = output_test.dtype
    batch_size = output_test.shape[0]
    pred_len = output_test.shape[1]

    # emission index: indicates which dimension of latent state is observable
    if emi_idx is None:
        emi_idx = [0]
    assert (len(emi_idx) == model.output_dim)

    # turn cov_g to be tensor: TODO: use likelihood module to parameterize cov_g
    cov_g = torch.tensor(cov_g, dtype=dtype).to(device)

    ''' -----------------  1.  using recognition network to get latent state 
    can also obtain the first prediction latent state by using the last state got from observable sequence
     -----------------'''
    if x_0 is None:
        q_x_0 = model.recognet(output_test, input_test)   # shape: batch_size x dim_state
        x_0 = q_x_0.rsample(torch.Size([N_MC]))           # shape: N_MC x batch_size x dim_state

    # initialization:
    pred_likelihood = torch.tensor(0.).to(device)
    y_pred_all = []
    x_t_1 = x_0

    if model._get_name() == 'JOGPSSM' or model._get_name() == 'COGPSSM' or  model._get_name() == 'GPSSM' :

        for t in range(pred_len):
            gp_input = x_t_1                                 # shape: N_MC x batch_size x state_dim
            if input_test is not None:
                c_t = input_test[:, t].repeat(N_MC, 1, 1)    # shape: N_MC x batch_size x input_dim
                gp_input = torch.cat((c_t, x_t_1), dim=-1)   # shape: N_MC x batch_size x (input_dim + state_dim)

            tmp = gp_input.repeat(model.state_dim, 1, 1, 1)  # shape: state_dim x N_MC x batch_size x (input_dim + state_dim)
            tmp = tmp.transpose(1, 0)                        # shape: N_MC x state_dim x batch_size x (input_dim + state_dim)
            qf_t = model.transition(tmp)                     # get function distribution: N_MC x state_dim x batch_size
            qx_t = model.likelihood(qf_t)                    # get state distribution: N_MC x state_dim x batch_size

            x_t = qx_t.rsample().transpose(-1,-2)            # shape: N_MC x batch_size x state_dim

            # emission model
            yt_mean = x_t[:, :, emi_idx]
            output_covar = cov_g * torch.eye(model.output_dim).expand(N_MC, batch_size, model.output_dim,
                                                                      model.output_dim).to(device)
            pyt = MultivariateNormal(yt_mean, output_covar)          # shape:  N_MC x batch_size x output_dim

            y_tmp = output_test[:, t].expand(N_MC, batch_size, model.output_dim) # shape:  N_MC x batch_size x output_dim
            pred_likelihood = pred_likelihood + pyt.log_prob(y_tmp).mean().div(pred_len)  # average over particles and batch

            # update x_t_1
            x_t_1 = x_t

            # save prediction
            # y_pred.append(pyt.sample().view(self.N_MC, batch_size, self.output_dim))
            y_pred_all.append(yt_mean)

    elif  model._get_name() == 'JOTGPSSM' or model._get_name() == 'COTGPSSM':
        for t in range(pred_len):
            gp_input = x_t_1                                 # shape: N_MC x batch_size x state_dim
            if input_test is not None:
                c_t = input_test[:, t].repeat(N_MC, 1, 1)    # shape: N_MC x batch_size x input_dim
                gp_input = torch.cat((c_t, x_t_1), dim=-1)   # shape: N_MC x batch_size x (input_dim + state_dim)

            tmp = gp_input.repeat(model.state_dim, 1, 1, 1)  # shape: state_dim x N_MC x batch_size x (input_dim + state_dim)
            tmp = tmp.transpose(1, 0)                        # shape: N_MC x state_dim x batch_size x (input_dim + state_dim)
            qf_t = model.transition(tmp)                     # get function distribution: N_MC x state_dim x batch_size

            sd_Gaussian = MultivariateNormal(torch.zeros(N_MC, batch_size, model.state_dim, device=device),
                                             torch.eye(model.state_dim, device=device).repeat(N_MC, batch_size, 1, 1)
                                             )

            f_t = qf_t.mean.transpose(2, 1)            # shape: N_MC x batch_size x state_dim

            if model.separate_flow:
                f_t_K = torch.zeros_like(f_t, device=device)
                for idx, fl in enumerate(model.flow):
                    # warp the samples
                    f_t_K[:, :, idx] = fl(f_t[:, :, idx])
            else:
                # normalizing flows: f_t_K = G(f_t), where f_t shape: N_MC x batch_size x state_dim
                f_t_K = model.flow(f_t)


            # shape: N_MC x batch_size x state_dim
            x_t = f_t_K + model.likelihood.noise.sqrt().squeeze() * sd_Gaussian.rsample()

            # emission model
            yt_mean = x_t[:, :, emi_idx]
            output_covar = cov_g * torch.eye(model.output_dim).expand(N_MC, batch_size, model.output_dim,
                                                                      model.output_dim).to(device)
            pyt = MultivariateNormal(yt_mean, output_covar)          # shape:  N_MC x batch_size x output_dim

            y_tmp = output_test[:, t].expand(N_MC, batch_size, model.output_dim) # shape:  N_MC x batch_size x output_dim
            pred_likelihood = pred_likelihood + pyt.log_prob(y_tmp).mean().div(pred_len)  # average over particles and batch

            # update x_t_1
            x_t_1 = x_t

            # save prediction
            # y_pred.append(pyt.sample().view(self.N_MC, batch_size, self.output_dim))
            y_pred_all.append(yt_mean)

    elif  model._get_name() == 'JOTGPSSM_realNVP' or model._get_name() == 'COTGPSSM_realNVP':
        for t in range(pred_len):
            gp_input = x_t_1                                 # shape: N_MC x batch_size x state_dim
            if input_test is not None:
                c_t = input_test[:, t].repeat(N_MC, 1, 1)    # shape: N_MC x batch_size x input_dim
                gp_input = torch.cat((c_t, x_t_1), dim=-1)   # shape: N_MC x batch_size x (input_dim + state_dim)

            tmp = gp_input.repeat(model.state_dim, 1, 1, 1)  # shape: state_dim x N_MC x batch_size x (input_dim + state_dim)
            tmp = tmp.transpose(1, 0)                        # shape: N_MC x state_dim x batch_size x (input_dim + state_dim)
            qf_t = model.transition(tmp)                     # get function distribution: N_MC x state_dim x batch_size

            sd_Gaussian = MultivariateNormal(torch.zeros(N_MC, batch_size, model.state_dim, device=device),
                                             torch.eye(model.state_dim, device=device).repeat(N_MC, batch_size, 1, 1)
                                             )

            f_t = qf_t.mean.transpose(2, 1)            # shape: N_MC x batch_size x state_dim

            f_t_K = f_t                                # shape: N_MC x batch_size x state_dim
            # flow transformations (using RealNVP)
            for flow in model.flow:
                f_t_K, _ = flow(f_t_K)

            # shape: N_MC x batch_size x state_dim
            x_t = f_t_K + model.likelihood.noise.sqrt().squeeze() * sd_Gaussian.rsample()

            # emission model
            yt_mean = x_t[:, :, emi_idx]
            output_covar = cov_g * torch.eye(model.output_dim).expand(N_MC, batch_size, model.output_dim,
                                                                      model.output_dim).to(device)
            pyt = MultivariateNormal(yt_mean, output_covar)          # shape:  N_MC x batch_size x output_dim

            y_tmp = output_test[:, t].expand(N_MC, batch_size, model.output_dim) # shape:  N_MC x batch_size x output_dim
            pred_likelihood = pred_likelihood + pyt.log_prob(y_tmp).mean().div(pred_len)  # average over particles and batch

            # update x_t_1
            x_t_1 = x_t

            # save prediction
            # y_pred.append(pyt.sample().view(self.N_MC, batch_size, self.output_dim))
            y_pred_all.append(yt_mean)

    else:
        raise NotImplementedError("Model {} not implemented.".format(model._get_name()))


    """ -----------------   globally for all the post-predictions    --------------------"""
    y_pred_all = torch.stack(y_pred_all, dim=0)     # shape: seq_len x N_MC x batch_size x output_dim
    y_pred_mean_all = y_pred_all.mean(dim=1)        # shape: seq_len x batch_size x output_dim
    y_pred_var_all = y_pred_all.std(dim=1)**2 + cov_g.view(-1, model.output_dim)  # shape: seq_len x batch_size x output_dim
    y_pred_std_all = torch.sqrt(y_pred_var_all)      # shape: seq_len x batch_size x output_dim

    MSE = 1 / pred_len * torch.norm(output_test[0, :, :] - y_pred_mean_all[:, 0, :]) ** 2
    print(f"MSE: {MSE}")
    RMSE = MSE.sqrt()

    output_test_original = output_test.cpu() * data_sd + data_mean       # shape: seq_len x batch_size x output_dim
    y_pred_mean_original = y_pred_mean_all.cpu() * data_sd + data_mean   # shape: seq_len x batch_size x output_dim
    MSE_original = 1 / pred_len * torch.norm(output_test_original[0, :, :] - y_pred_mean_original[:, 0, :]) ** 2
    print(f"original MSE : {MSE_original}")

    Y_pred_var_original = y_pred_std_all.cpu() * data_sd * data_sd          # shape: seq_len x batch_size x output_dim
    lower = y_pred_mean_original - 1.96 * torch.sqrt(Y_pred_var_original)   # shape: seq_len x batch_size x output_dim
    upper = y_pred_mean_original + 1.96 * torch.sqrt(Y_pred_var_original)   # shape: seq_len x batch_size x output_dim

    lower, upper = lower.view(-1, ).cpu().numpy(), upper.cpu().view(-1, ).numpy()

    f, ax = plt.subplots(1, 1)
    T = output_test_original.view(-1, 1).shape[0]
    plt.plot(range(T), output_test_original.view(-1, ), 'k-', label='true observations')
    plt.plot(range(T), y_pred_mean_original.view(-1, ), 'b-', label='predicted observations')
    ax.fill_between(range(T), lower, upper, color="b", alpha=0.2, label='95% CI')
    ax.legend(loc=0)  # , fontsize=28)
    plt.title(f'RMSE: {round(MSE_original.sqrt().item(), 3)}')
    if plt_save:
        plt.savefig(result_dir + f"prediction_performance_epoch{epoch}.pdf")
    plt.show()
    plt.close()

    return pred_likelihood, MSE, RMSE, MSE_original, MSE_original.sqrt()

def test_save(train_loader, test_loader, model, device, observation_noise_sd, result_dir, data_name, epoch):
    data_sd = test_loader.dataset.output_normalizer.sd
    data_mean = test_loader.dataset.output_normalizer.mean
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        if train_loader is None:
            MSE = []
            RMSE = []
            MSE_o = []
            RMSE_o = []
            ll = []
            for i_iter, (U_test, Y_test) in enumerate(test_loader):
                pred_ll, pred_MSE, pred_RMSE, \
                pred_MSE_o, pred_RMSE_o = predictive_observations(model, Y_test.to(device),
                                                                  cov_g=observation_noise_sd ** 2,
                                                                  data_sd=data_sd, data_mean=data_mean,
                                                                  input_test=U_test.to(device),emi_idx=None,
                                                                  result_dir=result_dir, epoch=epoch)
                MSE.append(pred_MSE)
                RMSE.append(pred_RMSE)
                MSE_o.append(pred_MSE_o)
                RMSE_o.append(pred_RMSE_o)
                ll.append(pred_ll)

            # Save the pred_LL, MSE, RMSE, to a CSV file
            df = {}
            df['MSE'] = torch.tensor(MSE).mean().item()
            df['RMSE'] = torch.tensor(RMSE).mean().item()
            df['MSE original'] = torch.tensor(MSE_o).mean().item()
            df['RMSE original'] = torch.tensor(RMSE_o).mean().item()
            df['Likelihood'] = torch.tensor(ll).mean().item()
            df = pd.DataFrame(df, index=[0])
            df.to_csv(result_dir + f"{data_name}_epoch{epoch}.csv", index=False)

            print("-" * 50)

        else:
            MSE = []
            RMSE = []
            ll = []
            for i_iter, (U_train, Y_train) in enumerate(train_loader):
                pred_ll, pred_MSE, pred_RMSE, \
                pred_MSE_o, pred_RMSE_o = predictive_observations(model, Y_train.to(device),
                                                                  cov_g=observation_noise_sd ** 2, data_sd=data_sd,
                                                                  data_mean=data_mean, input_test=U_train.to(device),
                                                                  emi_idx=None, epoch=epoch, result_dir=result_dir)
                MSE.append(pred_MSE)
                RMSE.append(pred_RMSE)
                ll.append(pred_ll)

            # Save the pred_LL, MSE, RMSE, to a CSV file
            df = {}
            df['train MSE'] = torch.tensor(MSE).mean().item()
            df['train RMSE'] = torch.tensor(RMSE).mean().item()
            df['train Likelihood'] = torch.tensor(ll).mean().item()
            print(f"train MSE {torch.tensor(MSE).mean().item()}")
            print(f"train RMSE {torch.tensor(RMSE).mean().item()}")
            print(f"train Likelihood {torch.tensor(ll).mean().item()}")
            print("-" * 50)

            MSE = []
            RMSE = []
            MSE_o = []
            RMSE_o = []
            ll = []
            for i_iter, (U_test, Y_test) in enumerate(test_loader):
                pred_ll, pred_MSE, pred_RMSE,\
                pred_MSE_o, pred_RMSE_o = predictive_observations(model, Y_test.to(device), cov_g=observation_noise_sd ** 2,
                                                                  data_sd=data_sd, data_mean=data_mean,
                                                                  input_test=U_test.to(device), emi_idx=None,
                                                                  epoch=epoch, result_dir=result_dir)
                MSE.append(pred_MSE)
                RMSE.append(pred_RMSE)
                MSE_o.append(pred_MSE_o)
                RMSE_o.append(pred_RMSE_o)
                ll.append(pred_ll)

            # Save the pred_LL, MSE, RMSE, to a CSV file
            df = {}
            df['MSE'] = torch.tensor(MSE).mean().item()
            df['RMSE'] = torch.tensor(RMSE).mean().item()
            df['MSE original'] = torch.tensor(MSE_o).mean().item()
            df['RMSE original'] = torch.tensor(RMSE_o).mean().item()
            df['Likelihood'] = torch.tensor(ll).mean().item()
            df = pd.DataFrame(df, index=[0])
            df.to_csv(result_dir + f"{data_name}_epoch{epoch}.csv", index=False)

            print("-" * 50)

