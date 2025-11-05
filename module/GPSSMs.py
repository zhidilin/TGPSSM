"""
#0. q(x_t) are directly parameterized by variational distribution, instead of using RNN
1. q(x_t) are directly parameterized by a LSTM-based inference network.
2. for Multidimensional latent state case
3. use recognition network and inference network
4. stochastic gradient descent
"""
import math
import torch
import torch.nn as nn
import gpytorch
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import TriangularLazyTensor
from .GPModels import IndependentMultitaskGPModel
from .recognition_network import LSTMRecognition
from .inference_network import MFInference, LSTMencoder, PostNet
from .utils import KL_divergence

def set_requires_grad(module, flag):
    """
    Sets requires_grad flag of all parameters of a torch.nn.module
    :param module: torch.nn.module
    :param flag: Flag to set requires_grad to
    """
    for param in module.parameters():
        param.requires_grad = flag

class SSM(nn.Module):
    def __init__(self, state_dim, output_dim, seq_len, inducing_points, input_dim=0, process_noise_sd=0.05, N_MC = 5):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.N_MC = N_MC
        # 定义GP transition
        self.transition = IndependentMultitaskGPModel(inducing_points=inducing_points, dim_state=self.state_dim)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([self.state_dim]))
        init_noise = process_noise_sd ** 2
        self.likelihood.noise = init_noise
        self.likelihood.requires_grad_(False)


class GPSSM(SSM):
    """
    Scalable learning using the structured inference network from paper:
    Eleftheriadis, Stefanos, et al.
            "Identification of Gaussian process state space models."
            Advances in neural information processing systems 30 (2017).

    It turns out that the training results easily fall into a bad local optimum,
    and it is quite hard to train the inference network.

    Joint Gaussian distribution for q(x_{0:T})
    """
    def __init__(self, state_dim, output_dim, seq_len, inducing_points, input_dim=0, process_noise_sd=0.05, N_MC = 5):
        super().__init__(state_dim, output_dim, seq_len, inducing_points, input_dim, process_noise_sd, N_MC)

        # define recognition network for initial state x_0
        self.recognet = LSTMRecognition(dim_outputs=self.output_dim,
                                        dim_inputs=self.input_dim,
                                        dim_states=self.state_dim,
                                        length=self.seq_len)

        # define inference network for learning the variational distribution of x_{1:T}
        self.infernet = MFInference(dim_outputs=self.output_dim,
                                    dim_inputs=self.input_dim,
                                    dim_states=self.state_dim,
                                    length=self.seq_len,
                                    hidden_size=32,
                                    num_layers=2,
                                    batch_first=True,
                                    bd=True)

    def forward(self, obsr, cov_g, input_sequence=None, emi_idx=None):

        dtype = obsr.dtype
        device = obsr.device
        batch_size = obsr.shape[0]

        # emission index: indicates which dimension of latent state is observable
        if emi_idx is None:
            emi_idx = [0]
        assert (len(emi_idx) == self.output_dim)

        # obsr.shape == [ batch_size x seq_len x output_dim ]
        assert (obsr.shape[-2] == self.seq_len)

        # 转 cov_g 为tensor
        cov_g = torch.tensor(cov_g, dtype=dtype).to(device)

        ###############    求 KL[ q(x0) || p(x0) ]   ###############
        # qx0
        qx0 = self.recognet(obsr, input_sequence)

        # 构建 px0
        px0_mean = torch.zeros_like(qx0.mean)                   # shape: batch_size x dim_state
        px0_cov = torch.diag_embed(torch.ones_like(qx0.mean))   # shape: batch_size x dim_state

        px0 = MultivariateNormal(px0_mean, px0_cov)
        qm0_KL = KL_divergence(qx0, px0).mean()                 # take average over batch_size

        ###############    求 A[1:T] 和 L[1:T]   ###############
        # At_all shape: [ batch_size x seq_len x dim_state x dim_state ]
        # Lt_all shape: [ batch_size x seq_len x dim_state x dim_state ]
        At_all, Lt_all = self.infernet(output_sequence=obsr, input_sequence=input_sequence)


        ################   calculate H(x_{1:T})    ###############
        const = self.seq_len * self.state_dim
        Hx = 0.5 * const * math.log(2 * math.pi) + torch.logdet(Lt_all).sum(dim=1).mean() + const * 0.5
        Hx = Hx.div(self.seq_len)

        #################  求高斯过程动态转移项 和 数据拟合项  ###############
        result_gp_dynamic = torch.tensor(0.).to(device)
        data_fit = torch.tensor(0.).to(device)

        # 采样x0:   shape: batch_size x dim_state
        xt_previous = qx0.rsample()

        mt_previous = qx0.mean.unsqueeze(dim=-1)             # shape: batch_size x dim_state x 1
        sigma_t_previous = qx0.covariance_matrix             # shape: batch_size x dim_state x dim_state

        for t in range(self.seq_len):
            ######  t 时刻观测值 对应的 隐状态 边缘分布
            mt =  At_all[:, t].matmul(mt_previous)                    # mt shape: batch_size x dim_state x 1
            _tmp = At_all[:, t].matmul(sigma_t_previous)              # _tmp shape: batch_size x dim_state x dim_state
            sigma_t = _tmp.matmul(At_all[:, t].transpose(-1,-2)) + Lt_all[:,t].matmul(Lt_all[:,t].transpose(-1,-2))
            sigma_t = sigma_t + 0.01 * torch.eye(self.state_dim, device=device).repeat(batch_size, 1, 1)
            qxt = MultivariateNormal(mt.squeeze(dim=-1), sigma_t)     # shape: batch_size x dim_state  

            mu_t = mt.squeeze(dim=-1)[:, emi_idx]
            V_t = sigma_t[:, emi_idx, emi_idx]
            V_t = V_t.view(batch_size, self.output_dim, self.output_dim)

            #####   t 时刻的 数据拟合 （基于观测yt）
            yt = obsr[:, t]         # shape: batch_size x dim_output
            obser_covar = cov_g * torch.eye(self.output_dim).expand(batch_size, self.output_dim, self.output_dim).to(device)
            y_dist = MultivariateNormal(mu_t, obser_covar)
            data_fit_tmp = y_dist.log_prob(yt).view(-1) - 0.5 * torch.diagonal(obser_covar.inverse().matmul(V_t),
                                                                               dim1=-1,
                                                                               dim2=-2).sum(dim=1)
            data_fit = data_fit + data_fit_tmp.mean()

            ''' ---------------      GP dynamics term          --------------- '''
            procs_noise = torch.diag_embed(self.likelihood.noise.view(-1),).expand(batch_size, self.state_dim, self.state_dim)

            # x_t_1 shape: batch_size x dim_state.     GP dynamics, shape: dim_state x batch_size
            gp_input = xt_previous      # shape:  batch_size x dim_state
            x_t = qxt.rsample()         # shape:  batch_size x dim_state
            if input_sequence is not None:
                c_t = input_sequence[:, t]                                   # shape: batch_size x dim_input
                gp_input = torch.cat((c_t, xt_previous), dim=-1)             # shape: batch_size x (dim_input + dim_state)
            gp_dynamics = self.transition( gp_input.expand(self.state_dim, batch_size, (self.state_dim+self.input_dim)) )
            x_pred = MultivariateNormal(gp_dynamics.mean.transpose(-2, -1), procs_noise)
            _result_gp_dynamic1 = x_pred.log_prob(x_t)
            _result_gp_dynamic2 = 1 / self.likelihood.noise.view(-1, self.state_dim) * gp_dynamics.variance.view(-1, self.state_dim)
            _result_gp_dynamic2 = - 0.5 * _result_gp_dynamic2.sum(-1)

            _result_gp_dynamic = _result_gp_dynamic1.view(-1) + _result_gp_dynamic2
            result_gp_dynamic = result_gp_dynamic + _result_gp_dynamic.mean().div(self.seq_len)

            # update:
            # shape: batch_size x dim_state
            xt_previous = x_t
            mt_previous = mt
            sigma_t_previous = sigma_t

            #求高斯过程的 KL 散度
        KL_div = self.transition.kl_divergence().div(self.seq_len)
        print()
        print(f" Entropy term: {Hx.item()}")
        print(f" x0 KL: {qm0_KL.item()} ")
        print(f" KL term: {KL_div} ")
        print(f" data-fit term: {data_fit.item()}")
        print(f" GP dynamic term: {result_gp_dynamic.item()}")
        print()
        return -qm0_KL + Hx - KL_div + data_fit + result_gp_dynamic

class JOGPSSM(SSM):
    """
    Scalable learning using the structured inference network (non-Gaussian variational distribution for q(x_{0:T})
    """
    def __init__(self, state_dim, output_dim, seq_len, inducing_points, input_dim=0,
                 process_noise_sd=0.05, N_MC=5, hidden_size=128, mf_flag=False):
        super().__init__(state_dim, output_dim, seq_len, inducing_points, input_dim, process_noise_sd, N_MC)

        # define recognition network for initial state x_0
        self.recognet = LSTMRecognition(dim_outputs=self.output_dim,
                                        dim_inputs=self.input_dim,
                                        dim_states=self.state_dim,
                                        length=self.seq_len)

        # encode y_{1:T} into a series of hidden states (Not to be confused, not latent states x_{1:T})
        self.encodeNet = LSTMencoder(dim_outputs=self.output_dim,
                                    dim_inputs=self.input_dim,
                                    dim_states=self.state_dim,
                                    length=self.seq_len,
                                    hidden_size=hidden_size,
                                    num_layers=2,
                                    batch_first=True,
                                    bd=True)

        # define inference network for q(x_{1:T})
        self.postNet = PostNet(x_dim=self.state_dim,
                               h_dim=self.encodeNet.hidden_size,
                               bd=self.encodeNet.bd,
                               mf_flag=mf_flag)

    def forward(self, obsr, cov_g, input_sequence=None, emi_idx=None):
        dtype = obsr.dtype
        device = obsr.device
        batch_size = obsr.shape[0]

        # emission index: indicates which dimension of latent state is observable
        if emi_idx is None:
            emi_idx = [0]
        assert (len(emi_idx) == self.output_dim)

        # obsr.shape == [ batch_size x seq_len x output_dim ]
        assert (obsr.shape[-2] == self.seq_len)

        # convert cov_g into a tensor
        cov_g = torch.tensor(cov_g, dtype=dtype).to(device)

        '''---------------   1.  KL[ q(x0) || p(x0) ]   --------------- '''
        # 得到 qx0
        qx0 = self.recognet(obsr, input_sequence)  # shape: batch_size x dim_state

        # 构建 px0
        px0_mean = torch.zeros_like(qx0.mean).to(device)                  # shape: batch_size x dim_state
        px0_cov = torch.diag_embed(torch.ones_like(qx0.mean)).to(device)  # shape: batch_size x dim_state

        px0 = MultivariateNormal(px0_mean, px0_cov)
        qm0_KL = KL_divergence(qx0, px0).mean()                           # take average over batch_size

        '''---------------   2.  encode y_{1:T} using LSTM   --------------- '''
        hidden_all = self.encodeNet(output_sequence=obsr, input_sequence=input_sequence)  # shape: batch_size x seq_len x (1+bd)*hidden_size

        '''---------------   initialization   --------------- '''
        result_gp_dynamic = torch.tensor(0.).to(device)
        data_fit = torch.tensor(0.).to(device)
        const = self.state_dim
        Hx = torch.tensor(0.).to(device) + const * 0.5 + 0.5 * const * math.log(2 * math.pi)

        x_t_1 = qx0.rsample()
        for t in range(self.seq_len):
            hidden_t = hidden_all[:, t]
            x_t, qxt = self.postNet(x_t_1=x_t_1, hidden=hidden_t)
            m_t, sigma_t = qxt.mean, qxt.covariance_matrix


            Hx_tmp =  0.5 * torch.logdet(sigma_t).mean()                 # average over batch_size
            Hx = Hx + Hx_tmp.div(self.seq_len)                           # calculate the entropy term

            '''---------------     data-fit (log-likelihood term)  based on y_t     --------------- '''

            yt = obsr[:, t]                                             # shape: batch_size x dim_output
            # observation covariance matrix
            obser_covar = cov_g * torch.eye(self.output_dim).expand(batch_size, self.output_dim, self.output_dim).to(device)

            # mean vector of the observable latent states
            mu_t = m_t[:, emi_idx]
            # covariance matrix of the observable latent states
            V_t = sigma_t[:, emi_idx]
            V_t = V_t[:, :, emi_idx]
            V_t = V_t.view(batch_size, self.output_dim, self.output_dim)

            # E_{q(x_t)} [ log p(y_t | x_t) ]
            y_dist = MultivariateNormal(mu_t, obser_covar)
            data_fit_tmp = y_dist.log_prob(yt).view(-1) - 0.5 * torch.diagonal(obser_covar.inverse().matmul(V_t),
                                                                               dim1=-1,
                                                                               dim2=-2).sum(dim=1)
            # data_fit_tmp = data_fit_tmp.div(self.seq_len)
            data_fit = data_fit + data_fit_tmp.mean()

            ''' ---------------      GP dynamics term          --------------- '''
            procs_noise = torch.diag_embed(self.likelihood.noise.view(-1),).expand(batch_size, self.state_dim, self.state_dim)

            # x_t_1 shape: batch_size x dim_state.     GP dynamics, shape: dim_state x batch_size
            gp_input = x_t_1      # shape:  batch_size x dim_state
            if input_sequence is not None:
                c_t = input_sequence[:, t]                             # shape: batch_size x dim_input
                gp_input = torch.cat((c_t, x_t_1), dim=-1)             # shape: batch_size x (dim_input + dim_state)
            gp_dynamics = self.transition( gp_input.expand(self.state_dim, batch_size, (self.state_dim+self.input_dim)) )
            x_pred = MultivariateNormal(gp_dynamics.mean.transpose(-2, -1), procs_noise)
            _result_gp_dynamic1 = x_pred.log_prob(x_t)
            _result_gp_dynamic2 = 1 / self.likelihood.noise.view(-1, self.state_dim) * gp_dynamics.variance.view(-1, self.state_dim)
            _result_gp_dynamic2 = - 0.5 * _result_gp_dynamic2.sum(-1)

            _result_gp_dynamic = _result_gp_dynamic1.view(-1) + _result_gp_dynamic2
            result_gp_dynamic = result_gp_dynamic + _result_gp_dynamic.mean().div(self.seq_len)

            # update x_t_1 for next round
            x_t_1 = x_t

        '''#################  Calculate the KL divergence term of variational GP  #################  '''
        KL_div = self.transition.kl_divergence().div(self.seq_len)

        print()
        print(f" Entropy term: {Hx.item()}")
        print(f" x0 KL: {qm0_KL.item()} ")
        print(f" KL term: {KL_div} ")
        print(f" data-fit term: {data_fit.item()}")
        print(f" GP dynamic term: {result_gp_dynamic.item()}")
        print()
        return -qm0_KL + Hx - KL_div + data_fit + result_gp_dynamic


class COGPSSM(SSM):
    """
    Scalable learning using the structured inference network (non-Gaussian variational distribution for q(x_{0:T})

    with constrained optimization framework
    """
    def __init__(self, state_dim, output_dim, seq_len, inducing_points, input_dim=0,
                 process_noise_sd=0.05, N_MC=5, hidden_size=128, mf_flag=False):
        super().__init__(state_dim, output_dim, seq_len, inducing_points, input_dim, process_noise_sd, N_MC)

        self.R0 = 0             # quality constraint for data-fitting and entropy term
        self.Rtmp = 0           # quality computed from last iteration
        self.beta = 1           # hyper-parameters for Lagrangian multiplier beta
        self.nu = 1e-3          # learning rate of beta
        self.alpha = 0.5        # hyperparameter of moving average for computing self.Rtmp

        # define recognition network for initial state x_0
        self.recognet = LSTMRecognition(dim_outputs=self.output_dim,
                                        dim_inputs=self.input_dim,
                                        dim_states=self.state_dim,
                                        length=self.seq_len)

        # encode y_{1:T} into a series of hidden states (Not to be confused, not latent states x_{1:T})
        self.encodeNet = LSTMencoder(dim_outputs=self.output_dim,
                                    dim_inputs=self.input_dim,
                                    dim_states=self.state_dim,
                                    length=self.seq_len,
                                    hidden_size=hidden_size,
                                    num_layers=2,
                                    batch_first=True,
                                    bd=True)

        # define inference network for q(x_{1:T})
        self.postNet = PostNet(x_dim=self.state_dim,
                               h_dim=self.encodeNet.hidden_size,
                               bd=self.encodeNet.bd,
                               mf_flag=mf_flag)

    def forward(self, obsr, cov_g, input_sequence=None, emi_idx=None):
        dtype = obsr.dtype
        device = obsr.device
        batch_size = obsr.shape[0]

        # emission index: indicates which dimension of latent state is observable
        if emi_idx is None:
            emi_idx = [0]
        assert (len(emi_idx) == self.output_dim)

        # obsr.shape == [ batch_size x seq_len x output_dim ]
        assert (obsr.shape[-2] == self.seq_len)

        # convert cov_g into a tensor
        cov_g = torch.tensor(cov_g, dtype=dtype).to(device)


        '''---------------   1.  KL[ q(x0) || p(x0) ]   --------------- '''
        qx0 = self.recognet(obsr, input_sequence)  # shape: batch_size x dim_state

        # construct p(x0)
        px0_mean = torch.zeros_like(qx0.mean).to(device)                  # shape: batch_size x dim_state
        px0_cov = torch.diag_embed(torch.ones_like(qx0.mean)).to(device)  # shape: batch_size x dim_state

        px0 = MultivariateNormal(px0_mean, px0_cov)
        qm0_KL = KL_divergence(qx0, px0).mean()                           # take average over batch_size

        '''---------------   2.  encode y_{1:T} using LSTM   --------------- '''
        hidden_all = self.encodeNet(output_sequence=obsr, input_sequence=input_sequence)  # shape: batch_size x seq_len x (1+bd)*hidden_size

        '''---------------   initialization   --------------- '''
        result_gp_dynamic = torch.tensor(0.).to(device)
        data_fit = torch.tensor(0.).to(device)
        const = self.state_dim
        Hx = torch.tensor(0.).to(device) + const * 0.5 + 0.5 * const * math.log(2 * math.pi)

        x_t_1 = qx0.rsample()
        for t in range(self.seq_len):
            hidden_t = hidden_all[:, t]
            x_t, qxt = self.postNet(x_t_1=x_t_1, hidden=hidden_t)
            m_t, sigma_t = qxt.mean, qxt.covariance_matrix


            Hx_tmp =  0.5 * torch.logdet(sigma_t).mean()                 # average over batch_size
            Hx = Hx + Hx_tmp.div(self.seq_len)                           # calculate the entropy term

            '''---------------     data-fit (log-likelihood term)  based on y_t     --------------- '''

            yt = obsr[:, t]                                             # shape: batch_size x dim_output
            # observation covariance matrix
            obser_covar = cov_g * torch.eye(self.output_dim).expand(batch_size, self.output_dim, self.output_dim).to(device)

            # mean vector of the observable latent states
            mu_t = m_t[:, emi_idx]
            # covariance matrix of the observable latent states
            V_t = sigma_t[:, emi_idx]
            V_t = V_t[:, :, emi_idx]
            V_t = V_t.view(batch_size, self.output_dim, self.output_dim)

            # E_{q(x_t)} [ log p(y_t | x_t) ]
            y_dist = MultivariateNormal(mu_t, obser_covar)
            data_fit_tmp = y_dist.log_prob(yt).view(-1) - 0.5 * torch.diagonal(obser_covar.inverse().matmul(V_t),
                                                                               dim1=-1,
                                                                               dim2=-2).sum(dim=1)
            data_fit = data_fit + data_fit_tmp.mean().div(self.seq_len)

            ''' ---------------      GP dynamics term          --------------- '''
            procs_noise = torch.diag_embed(self.likelihood.noise.view(-1),).expand(batch_size, self.state_dim, self.state_dim)
            # x_t_1 shape: batch_size x dim_state.     GP dynamics, shape: dim_state x batch_size
            gp_input = x_t_1     # shape:  batch_size x dim_state
            if input_sequence is not None:
                c_t = input_sequence[:, t]                             # shape: batch_size x dim_input
                gp_input = torch.cat((c_t, x_t_1), dim=-1)             # shape: batch_size x (dim_input + dim_state)
            gp_dynamics = self.transition( gp_input.expand(self.state_dim, batch_size, (self.state_dim+self.input_dim)) )
            x_pred = MultivariateNormal(gp_dynamics.mean.transpose(-2, -1), procs_noise)
            _result_gp_dynamic1 = x_pred.log_prob(x_t)
            _result_gp_dynamic2 = 1 / self.likelihood.noise.view(-1, self.state_dim) * gp_dynamics.variance.view(-1, self.state_dim)
            _result_gp_dynamic2 = - 0.5 * _result_gp_dynamic2.sum(-1)

            _result_gp_dynamic = _result_gp_dynamic1.view(-1) + _result_gp_dynamic2
            result_gp_dynamic = result_gp_dynamic + _result_gp_dynamic.mean().div(self.seq_len)

            # update x_t_1 for next round
            x_t_1 = x_t

        '''#################  Calculate the KL divergence term of variational GP  #################  '''
        KL_div = self.transition.kl_divergence().div(self.seq_len)

        print()
        print(f" Entropy term: {Hx.item()}")
        print(f" x0 KL: {qm0_KL.item()} ")
        print(f" KL term: {KL_div} ")
        print(f" data-fit term: {data_fit.item()}")
        print(f" GP dynamic term: {result_gp_dynamic.item()}")
        print(f"ELBO: {(-qm0_KL + Hx + data_fit) + result_gp_dynamic - KL_div}")

        '''------------------------    update beta    ------------------------'''
        self.Rtmp  = (1 - self.alpha) * self.Rtmp + self.alpha * (data_fit.detach())
        self.beta = self.beta * math.exp(self.nu * (self.R0 - self.Rtmp))
        print(f"beta: {self.beta}")
        print()

        return -qm0_KL + Hx + self.beta * (data_fit) + result_gp_dynamic - KL_div



    def calculate_r0(self, obsr, cov_g, input_sequence=None, emi_idx=None):
        dtype = obsr.dtype
        device = obsr.device
        batch_size = obsr.shape[0]

        # emission index: indicates which dimension of latent state is observable
        if emi_idx is None:
            emi_idx = [0]
        assert (len(emi_idx) == self.output_dim)

        # obsr.shape == [ batch_size x seq_len x output_dim ]
        assert (obsr.shape[-2] == self.seq_len)

        # convert cov_g into a tensor
        cov_g = torch.tensor(cov_g, dtype=dtype).to(device)


        '''---------------   1.  KL[ q(x0) || p(x0) ]   --------------- '''
        qx0 = self.recognet(obsr, input_sequence)  # shape: batch_size x dim_state

        # construct p(x0)
        px0_mean = torch.zeros_like(qx0.mean).to(device)                  # shape: batch_size x dim_state
        px0_cov = torch.diag_embed(torch.ones_like(qx0.mean)).to(device)  # shape: batch_size x dim_state

        px0 = MultivariateNormal(px0_mean, px0_cov)
        qm0_KL = KL_divergence(qx0, px0).mean()                           # take average over batch_size

        '''---------------   2.  encode y_{1:T} using LSTM   --------------- '''
        hidden_all = self.encodeNet(output_sequence=obsr, input_sequence=input_sequence)  # shape: batch_size x seq_len x (1+bd)*hidden_size

        '''---------------   initialization   --------------- '''
        data_fit = torch.tensor(0.).to(device)
        const = self.state_dim
        Hx = torch.tensor(0.).to(device) + const * 0.5 + 0.5 * const * math.log(2 * math.pi)

        KL_div = torch.tensor(0.).to(device)
        result_gp_dynamic = torch.tensor(0.).to(device)

        x_t_1 = qx0.rsample()
        for t in range(self.seq_len):
            hidden_t = hidden_all[:, t]
            x_t, qxt = self.postNet(x_t_1=x_t_1, hidden=hidden_t)
            m_t, sigma_t = qxt.mean, qxt.covariance_matrix

            Hx_tmp =  0.5 * torch.logdet(sigma_t).mean()                 # average over batch_size
            Hx = Hx + Hx_tmp.div(self.seq_len)                           # calculate the entropy term

            '''---------------     data-fit (log-likelihood term)  based on y_t     --------------- '''

            yt = obsr[:, t]                                             # shape: batch_size x dim_output
            # observation covariance matrix
            obser_covar = cov_g * torch.eye(self.output_dim).expand(batch_size, self.output_dim, self.output_dim).to(device)

            # mean vector of the observable latent states
            mu_t = m_t[:, emi_idx]
            # covariance matrix of the observable latent states
            V_t = sigma_t[:, emi_idx]
            V_t = V_t[:, :, emi_idx]
            V_t = V_t.view(batch_size, self.output_dim, self.output_dim)

            # E_{q(x_t)} [ log p(y_t | x_t) ]
            y_dist = MultivariateNormal(mu_t, obser_covar)
            data_fit_tmp = y_dist.log_prob(yt).view(-1) - 0.5 * torch.diagonal(obser_covar.inverse().matmul(V_t),
                                                                               dim1=-1,
                                                                               dim2=-2).sum(dim=1)
            data_fit = data_fit + data_fit_tmp.mean().div(self.seq_len)

            # update x_t_1 for next round
            x_t_1 = x_t

        print(f" Entropy term: {Hx.item()}")
        print(f" x0 KL: {qm0_KL.item()} ")
        print(f" KL term: {KL_div} ")
        print(f" data-fit term: {data_fit.item()}")
        print(f" GP dynamic term: {result_gp_dynamic.item()}")
        print()
        return -qm0_KL + Hx - KL_div + data_fit + result_gp_dynamic, data_fit


# class COGPSSM_old(SSM):
#     """
#     Scalable learning using the structured inference network (non-Gaussian variational distribution for q(x_{0:T})
#
#     with constrained optimization framework
#     """
#     def __init__(self, state_dim, output_dim, seq_len, inducing_points, input_dim=0,
#                  process_noise_sd=0.05, N_MC=5, hidden_size=128, mf_flag=False, tau1=10, tau2=0.1, nu=300):
#         super().__init__(state_dim, output_dim, seq_len, inducing_points, input_dim, process_noise_sd, N_MC)
#
#         # quality constraint for data-fitting and entropy term
#         self.R0 = 0
#         # hyper-parameters for Lagrangian multiplier beta
#         self.beta = 1
#         self.tau1=tau1
#         self.tau2=tau2
#         self.nu=nu
#
#         # define recognition network for initial state x_0
#         self.recognet = LSTMRecognition(dim_outputs=self.output_dim,
#                                         dim_inputs=self.input_dim,
#                                         dim_states=self.state_dim,
#                                         length=self.seq_len)
#
#         # encode y_{1:T} into a series of hidden states (Not to be confused, not latent states x_{1:T})
#         self.encodeNet = LSTMencoder(dim_outputs=self.output_dim,
#                                     dim_inputs=self.input_dim,
#                                     dim_states=self.state_dim,
#                                     length=self.seq_len,
#                                     hidden_size=hidden_size,
#                                     num_layers=2,
#                                     batch_first=True,
#                                     bd=True)
#
#         # define inference network for q(x_{1:T})
#         self.postNet = PostNet(x_dim=self.state_dim,
#                                h_dim=self.encodeNet.hidden_size,
#                                bd=self.encodeNet.bd,
#                                mf_flag=mf_flag)
#
#     def forward(self, obsr, cov_g, input_sequence=None, emi_idx=None):
#         dtype = obsr.dtype
#         device = obsr.device
#         batch_size = obsr.shape[0]
#
#         # emission index: indicates which dimension of latent state is observable
#         if emi_idx is None:
#             emi_idx = [0]
#         assert (len(emi_idx) == self.output_dim)
#
#         # obsr.shape == [ batch_size x seq_len x output_dim ]
#         assert (obsr.shape[-2] == self.seq_len)
#
#         # convert cov_g into a tensor
#         cov_g = torch.tensor(cov_g, dtype=dtype).to(device)
#
#
#         '''---------------   1.  KL[ q(x0) || p(x0) ]   --------------- '''
#         qx0 = self.recognet(obsr, input_sequence)  # shape: batch_size x dim_state
#
#         # construct px0
#         px0_mean = torch.zeros_like(qx0.mean).to(device)                  # shape: batch_size x dim_state
#         px0_cov = torch.diag_embed(torch.ones_like(qx0.mean)).to(device)  # shape: batch_size x dim_state
#
#         px0 = MultivariateNormal(px0_mean, px0_cov)
#         qm0_KL = KL_divergence(qx0, px0).mean()                           # take average over batch_size
#
#         '''---------------   2.  encode y_{1:T} using LSTM   --------------- '''
#         hidden_all = self.encodeNet(output_sequence=obsr, input_sequence=input_sequence)  # shape: batch_size x seq_len x (1+bd)*hidden_size
#
#         '''---------------   initialization   --------------- '''
#         result_gp_dynamic = torch.tensor(0.).to(device)
#         data_fit = torch.tensor(0.).to(device)
#         const = self.state_dim
#         Hx = torch.tensor(0.).to(device) + const * 0.5 + 0.5 * const * math.log(2 * math.pi)
#
#         x_t_1 = qx0.rsample()
#         for t in range(self.seq_len):
#             hidden_t = hidden_all[:, t]
#             x_t, qxt = self.postNet(x_t_1=x_t_1, hidden=hidden_t)
#             m_t, sigma_t = qxt.mean, qxt.covariance_matrix
#
#
#             Hx_tmp =  0.5 * torch.logdet(sigma_t).mean()                 # average over batch_size
#             Hx = Hx + Hx_tmp.div(self.seq_len)                           # calculate the entropy term
#
#             '''---------------     data-fit (log-likelihood term)  based on y_t     --------------- '''
#
#             yt = obsr[:, t]                                             # shape: batch_size x dim_output
#             # observation covariance matrix
#             obser_covar = cov_g * torch.eye(self.output_dim).expand(batch_size, self.output_dim, self.output_dim).to(device)
#
#             # mean vector of the observable latent states
#             mu_t = m_t[:, emi_idx]
#             # covariance matrix of the observable latent states
#             V_t = sigma_t[:, emi_idx]
#             V_t = V_t[:, :, emi_idx]
#             V_t = V_t.view(batch_size, self.output_dim, self.output_dim)
#
#             # E_{q(x_t)} [ log p(y_t | x_t) ]
#             y_dist = MultivariateNormal(mu_t, obser_covar)
#             data_fit_tmp = y_dist.log_prob(yt).view(-1) - 0.5 * torch.diagonal(obser_covar.inverse().matmul(V_t),
#                                                                                dim1=-1,
#                                                                                dim2=-2).sum(dim=1)
#             data_fit = data_fit + data_fit_tmp.mean().div(self.seq_len)
#
#             ''' ---------------      GP dynamics term          --------------- '''
#             procs_noise = torch.diag_embed(self.likelihood.noise.view(-1),).expand(batch_size, self.state_dim, self.state_dim)
#             # x_t_1 shape: batch_size x dim_state.     GP dynamics, shape: dim_state x batch_size
#             gp_input = x_t_1.detach()      # shape:  batch_size x dim_state
#             if input_sequence is not None:
#                 c_t = input_sequence[:, t]                             # shape: batch_size x dim_input
#                 gp_input = torch.cat((c_t, x_t_1.detach()), dim=-1)             # shape: batch_size x (dim_input + dim_state)
#             gp_dynamics = self.transition( gp_input.expand(self.state_dim, batch_size, (self.state_dim+self.input_dim)) )
#             x_pred = MultivariateNormal(gp_dynamics.mean.transpose(-2, -1), procs_noise)
#             _result_gp_dynamic1 = x_pred.log_prob(x_t.detach())
#             _result_gp_dynamic2 = 1 / self.likelihood.noise.view(-1, self.state_dim) * gp_dynamics.variance.view(-1, self.state_dim)
#             _result_gp_dynamic2 = - 0.5 * _result_gp_dynamic2.sum(-1)
#
#             _result_gp_dynamic = _result_gp_dynamic1.view(-1) + _result_gp_dynamic2
#             result_gp_dynamic = result_gp_dynamic + _result_gp_dynamic.mean().div(self.seq_len)
#
#             # update x_t_1 for next round
#             x_t_1 = x_t
#
#         '''#################  Calculate the KL divergence term of variational GP  #################  '''
#         KL_div = self.transition.kl_divergence().div(self.seq_len)
#
#
#         R = (-qm0_KL + Hx + data_fit)
#         print()
#         print(f" Entropy term: {Hx.item()}")
#         print(f" x0 KL: {qm0_KL.item()} ")
#         print(f" KL term: {KL_div} ")
#         print(f" data-fit term: {data_fit.item()}")
#         print(f" GP dynamic term: {result_gp_dynamic.item()}")
#         print(f"ELBO: {R + result_gp_dynamic - KL_div}")
#
#         if R >= self.R0:
#             # Lambda = math.tanh(self.tau1 * (1-self.beta) / self.beta )
#             self.beta = 1.0 #self.beta * math.exp( -self.nu * Lambda * (-R.detach().cpu().numpy() + self.R0))
#             print(f"beta: {self.beta}")
#             print()
#
#         else:
#             self.beta = 30.0 #self.beta * math.exp( 1 + self.tau2 )
#             print(f"beta: {self.beta}")
#             print()
#
#         return self.beta * R + result_gp_dynamic - KL_div
#
#
#
#     def calculate_r0(self, obsr, cov_g, input_sequence=None, emi_idx=None):
#         dtype = obsr.dtype
#         device = obsr.device
#         batch_size = obsr.shape[0]
#
#         # emission index: indicates which dimension of latent state is observable
#         if emi_idx is None:
#             emi_idx = [0]
#         assert (len(emi_idx) == self.output_dim)
#
#         # obsr.shape == [ batch_size x seq_len x output_dim ]
#         assert (obsr.shape[-2] == self.seq_len)
#
#         # convert cov_g into a tensor
#         cov_g = torch.tensor(cov_g, dtype=dtype).to(device)
#
#
#         '''---------------   1.  KL[ q(x0) || p(x0) ]   --------------- '''
#         # 得到 qx0
#         qx0 = self.recognet(obsr, input_sequence)  # shape: batch_size x dim_state
#
#         # 构建 px0
#         px0_mean = torch.zeros_like(qx0.mean).to(device)                  # shape: batch_size x dim_state
#         px0_cov = torch.diag_embed(torch.ones_like(qx0.mean)).to(device)  # shape: batch_size x dim_state
#
#         px0 = MultivariateNormal(px0_mean, px0_cov)
#         qm0_KL = KL_divergence(qx0, px0).mean()                           # take average over batch_size
#
#         '''---------------   2.  encode y_{1:T} using LSTM   --------------- '''
#         hidden_all = self.encodeNet(output_sequence=obsr, input_sequence=input_sequence)  # shape: batch_size x seq_len x (1+bd)*hidden_size
#
#         '''---------------   initialization   --------------- '''
#         data_fit = torch.tensor(0.).to(device)
#         const = self.state_dim
#         Hx = torch.tensor(0.).to(device) + const * 0.5 + 0.5 * const * math.log(2 * math.pi)
#
#         KL_div = torch.tensor(0.).to(device)
#         result_gp_dynamic = torch.tensor(0.).to(device)
#
#         x_t_1 = qx0.rsample()
#         for t in range(self.seq_len):
#             hidden_t = hidden_all[:, t]
#             x_t, qxt = self.postNet(x_t_1=x_t_1, hidden=hidden_t)
#             m_t, sigma_t = qxt.mean, qxt.covariance_matrix
#
#             Hx_tmp =  0.5 * torch.logdet(sigma_t).mean()                 # average over batch_size
#             Hx = Hx + Hx_tmp.div(self.seq_len)                           # calculate the entropy term
#
#             '''---------------     data-fit (log-likelihood term)  based on y_t     --------------- '''
#
#             yt = obsr[:, t]                                             # shape: batch_size x dim_output
#             # observation covariance matrix
#             obser_covar = cov_g * torch.eye(self.output_dim).expand(batch_size, self.output_dim, self.output_dim).to(device)
#
#             # mean vector of the observable latent states
#             mu_t = m_t[:, emi_idx]
#             # covariance matrix of the observable latent states
#             V_t = sigma_t[:, emi_idx]
#             V_t = V_t[:, :, emi_idx]
#             V_t = V_t.view(batch_size, self.output_dim, self.output_dim)
#
#             # E_{q(x_t)} [ log p(y_t | x_t) ]
#             y_dist = MultivariateNormal(mu_t, obser_covar)
#             data_fit_tmp = y_dist.log_prob(yt).view(-1) - 0.5 * torch.diagonal(obser_covar.inverse().matmul(V_t),
#                                                                                dim1=-1,
#                                                                                dim2=-2).sum(dim=1)
#             data_fit = data_fit + data_fit_tmp.mean().div(self.seq_len)
#
#             # update x_t_1 for next round
#             x_t_1 = x_t
#
#         print(f" Entropy term: {Hx.item()}")
#         print(f" x0 KL: {qm0_KL.item()} ")
#         print(f" KL term: {KL_div} ")
#         print(f" data-fit term: {data_fit.item()}")
#         print(f" GP dynamic term: {result_gp_dynamic.item()}")
#         print()
#         return -qm0_KL + Hx - KL_div + data_fit + result_gp_dynamic
