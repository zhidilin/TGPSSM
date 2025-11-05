# Updated version of TGPSSMs_new.py:
#       Use quadrature point to compute the TGP transition expectation:   E_q(x_{t-1}, f_t) [x_t | G(f_t) ]
#
#       TGPSSM modules: construct to use normalizing flow models for each dimension using elementary flows
#
"""
# 0. q(x_t) are directly parameterized by variational distribution, instead of using RNN
1. q(x_t) are directly parameterized by a LSTM-based inference network.
2. for Multidimensional latent state case
3. use recognition network and inference network
4. stochastic gradient descent
"""
import math
import torch
import copy
import torch.nn as nn
import gpytorch
from gpytorch.distributions import MultivariateNormal
from .GPSSMs import SSM
from .GPModels import IndependentMultitaskGPModel, GaussianNonLinearMean, ELL
from .recognition_network import LSTMRecognition
from .inference_network import LSTMencoder, PostNet
from .utils import KL_divergence
from .Flows import instance_flow, SAL, initialize_flows, TANH

class JOTGPSSM_E(SSM):
    # TGPSSM with elementary flows, especially for the 1-D latent state-space case
    def __init__(self, state_dim, output_dim, seq_len, inducing_points, input_dim=0,
                 process_noise_sd=0.03,
                 N_MC=50,  # number of particle to sample f_t from q(f_t)
                 num_flow_blocks = 3,
                 hidden_size=128,
                 mf_flag=False,
                 separate_flow=True):
        super().__init__(state_dim, output_dim, seq_len, inducing_points, input_dim, process_noise_sd, N_MC)
        self.num_blocks = num_flow_blocks # define number of blocks for normalizing flow

        # GP transition
        self.transition = IndependentMultitaskGPModel(inducing_points=inducing_points, dim_state=self.state_dim)
        self.likelihood = GaussianNonLinearMean(out_dim=self.output_dim,  noise_init=process_noise_sd**2,
                                                noise_is_shared=True, quadrature_points=100)
        # self.likelihood.requires_grad_(False)  # TODO: False

        # if the data is 1-D synthetic data
        if self.state_dim == 1:
            # Initialize elementary normalizing flows
            flow = SAL(num_blocks=self.num_blocks)
            flow_tanh = TANH(num_blocks=1)  # default = 1
            for i in range(len(flow_tanh)):
                flow.append(flow_tanh[i])
            flow_SAL = SAL(num_blocks=1)    # default = 1
            flow.append(flow_SAL[0])

        # if the data is general data
        else:
            # Initialize elementary normalizing flows
            flow = SAL(num_blocks=self.num_blocks)
            flow_SAL = SAL(num_blocks=1)  # default = 1
            flow.append(flow_SAL[0])
            flow_tanh = TANH(num_blocks=2)  # default = 2
            for i in range(len(flow_tanh)):
                flow.append(flow_tanh[i])

        flow = instance_flow(flow)              # shared NF for each state dimension

        self.separate_flow = separate_flow      # separate NF for each state dimension (only for elementary flows)
        if self.separate_flow:
            flow = [copy.deepcopy(flow) for i in range(self.state_dim)]  # create one flow per state dimension,
            # need the copy.deepcopy so that each class has its own instance

        else:
            flow = [copy.deepcopy(flow) for i in range(1)]     # create only one flow each state dimension,
            # need the copy.deepcopy so that each class has its own instance

        self.flow = initialize_flows(flow)

        # recognition network for initial state x_0
        self.recognet = LSTMRecognition(dim_outputs=self.output_dim,
                                        dim_inputs=self.input_dim,
                                        dim_states=self.state_dim,
                                        length=self.seq_len)

        # encode y_{1:T} into hidden states
        self.encodeNet = LSTMencoder(dim_outputs=self.output_dim,
                                    dim_inputs=self.input_dim,
                                    dim_states=self.state_dim,
                                    length=self.seq_len,
                                    hidden_size=hidden_size,
                                    num_layers=2,
                                    batch_first=True,
                                    bd=True)

        # variational distribution q(x_{1:T})
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

        # change 'cov_g' into a tensor
        cov_g = torch.tensor(cov_g, dtype=dtype).to(device)

        ''' ---------------  1.  KL[ q(x0) || p(x0) ]  --------------- '''
        # qx0: shape: batch_size x dim_state
        qx0 = self.recognet(obsr, input_sequence)

        # construct px0
        px0_mean = torch.zeros_like(qx0.mean)                    # shape: batch_size x dim_state
        px0_cov = torch.diag_embed(torch.ones_like(qx0.mean))    # shape: batch_size x dim_state

        px0 = MultivariateNormal(px0_mean, px0_cov)
        qm0_KL = KL_divergence(qx0, px0).mean()                  # take average over batch_size

        '''---------------  2.  encode y_{1:T} using LSTM   --------------- '''
        hidden_all = self.encodeNet(output_sequence=obsr, input_sequence=input_sequence)  # shape: batch_size x seq_len x (1+bd)*hidden_size

        '''---------------  initialization   ---------------'''
        result_gp_dynamic = torch.tensor(0.).to(device)
        data_fit = torch.tensor(0.).to(device)
        const = self.state_dim
        Hx = torch.tensor(0.).to(device) + const * 0.5 + 0.5 * const * math.log(2 * math.pi)

        x_t_1 = qx0.rsample()       # shape: batch_size x dim_state
        for t in range(self.seq_len):
            hidden_t = hidden_all[:, t]
            x_t, qxt = self.postNet(x_t_1=x_t_1, hidden=hidden_t)        # shape: batch_size x dim_states
            sigma_t = qxt.covariance_matrix

            Hx_tmp =  0.5 * torch.logdet(sigma_t).mean()                 # average over batch_size (ignore the constant)
            Hx = Hx + Hx_tmp.div(self.seq_len)                           # calculate the entropy term

            '''---------------    data-fit (log-likelihood term)  based on y_t      ---------------'''
            yt = obsr[:, t]                                             # shape: batch_size x dim_output
            # observation covariance matrix
            obser_covar = cov_g * torch.eye(self.output_dim).expand(batch_size, self.output_dim, self.output_dim).to(device)

            # mean vector of the observable latent states
            mu_t = x_t[:, emi_idx]
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

            ''' ---------------      TGP dynamics term           ---------------'''
            # x_t_1 shape: batch_size x dim_state.     GP dynamics, shape: dim_state x batch_size
            gp_input = x_t_1   # shape:  batch_size x dim_state
            if input_sequence is not None:
                c_t = input_sequence[:, t]                  # shape: batch_size x dim_input
                gp_input = torch.cat((c_t, x_t_1), dim=-1)  # shape: batch_size x (dim_input + dim_state)

            # TODO: check if '_result_gp_dynamic' has been divided by 'seq_len'
            # ELL: 4th argument: [dim_state, MB, dim_input+dim_state], 5th argument: [dim_state, MB]
            _result_gp_dynamic = ELL(self.transition,
                                     self.flow,
                                     self.likelihood,
                                     gp_input.repeat(self.state_dim, 1, 1),
                                     x_t.transpose(0,1))     # should output a scalar

            result_gp_dynamic = result_gp_dynamic + _result_gp_dynamic.div(self.seq_len)


            # ''' ------ resample f_t from q(f_t) ------'''
            # qf_t = self.transition(gp_input.repeat(self.state_dim, 1, 1))   # shape: dim_states x batch_size
            # f_t = qf_t.rsample(torch.Size([self.N_MC])).transpose(-1, -2)   # shape: N_MC x batch_size x dim_states
            #
            # # flow transformations
            # if self.separate_flow:
            #     # use separate flow to model each state dimension
            #     f_k_t = torch.zeros_like(f_t, device=device)
            #     for idx, fl in enumerate(self.flow):
            #         # warp the samples
            #         f_k_t[:, :, idx] = fl(f_t[:, :, idx])
            # else:
            #     # shared flow
            #     f_k_t = self.flow(f_t)
            #
            # # process noise model
            # procs_noise = torch.diag_embed(self.likelihood.noise.view(-1), ).repeat(self.N_MC, batch_size, 1, 1)
            # qx_t_pre = MultivariateNormal(f_k_t, procs_noise)           # shape: N_MC x batch_size x dim_states
            #
            # # transition dynamics fitting
            # x_t_tmp = x_t.repeat(self.N_MC, 1, 1)                       # shape: N_MC x batch_size x dim_states
            # _result_gp_dynamic = qx_t_pre.log_prob(x_t_tmp)             # shape: N_MC x batch_size x 1
            # _result_gp_dynamic = _result_gp_dynamic.mean(dim=0)         # shape: batch_size x 1
            #
            # # _result_gp_dynamic
            # result_gp_dynamic = result_gp_dynamic + _result_gp_dynamic.mean().div(self.seq_len)
            # # print(f"result_gp_dynamic {t}: {_result_gp_dynamic.mean().div(self.seq_len)}")

            # update x_t_1 for next round
            x_t_1 = x_t

        '''---------------  Calculate the KL divergence term of variational GP  ---------------'''
        KL_div = self.transition.kl_divergence().div(self.seq_len)
        ELBO = -qm0_KL + Hx - KL_div + data_fit + result_gp_dynamic

        print()
        print(f" Entropy term: {Hx.item()}")
        print(f" x0 KL: {qm0_KL.item()} ")
        print(f" KL term: {KL_div} ")
        print(f" data-fit term: {data_fit.item()}")
        print(f" GP dynamic term: {result_gp_dynamic.item()}")
        print(f"ELBO: {ELBO.item()}")
        print()
        return ELBO

class COTGPSSM_E(SSM):
    # TGPSSM with elementary flows, especially for the 1-D latent state-space case
    def __init__(self, state_dim, output_dim, seq_len, inducing_points, input_dim=0,
                 process_noise_sd=0.03,
                 N_MC=50,  # number of particle to sample f_t from q(f_t)
                 num_flow_blocks = 3,
                 hidden_size=128,
                 mf_flag=False,
                 separate_flow=True):

        super().__init__(state_dim, output_dim, seq_len, inducing_points, input_dim, process_noise_sd, N_MC)
        self.num_blocks = num_flow_blocks  # define number of blocks for normalizing flow
        self.R0 = 0  # quality constraint for data-fitting and entropy term
        self.Rtmp = 0  # quality computed from last iteration
        self.beta = 1  # hyper-parameters for Lagrangian multiplier beta
        self.nu = 1e-3  # learning rate of beta
        self.alpha = 0.5  # hyperparameter of moving average for computing self.Rtmp

        # GP transition
        self.transition = IndependentMultitaskGPModel(inducing_points=inducing_points, dim_state=self.state_dim)
        self.likelihood = GaussianNonLinearMean(out_dim=self.output_dim,  noise_init=process_noise_sd**2,
                                                noise_is_shared=True, quadrature_points=100)
        # self.likelihood.requires_grad_(False)  # TODO: False

        # if the data is 1-D synthetic data
        if self.state_dim == 1:
            # Initialize elementary normalizing flows
            flow = SAL(num_blocks=self.num_blocks)
            flow_tanh = TANH(num_blocks=1)  # default = 1
            for i in range(len(flow_tanh)):
                flow.append(flow_tanh[i])
            flow_SAL = SAL(num_blocks=1)    # default = 1
            flow.append(flow_SAL[0])

        # if the data is general data
        else:
            # Initialize elementary normalizing flows
            flow = SAL(num_blocks=self.num_blocks)
            flow_SAL = SAL(num_blocks=1)  # default = 1
            flow.append(flow_SAL[0])
            flow_tanh = TANH(num_blocks=2)  # default = 2
            for i in range(len(flow_tanh)):
                flow.append(flow_tanh[i])

        flow = instance_flow(flow)              # shared NF for each state dimension

        self.separate_flow = separate_flow      # separate NF for each state dimension (only for elementary flows)
        if self.separate_flow:
            flow = [copy.deepcopy(flow) for i in range(self.state_dim)]  # create one flow per state dimension,
            # need the copy.deepcopy so that each class has its own instance

        else:
            flow = [copy.deepcopy(flow) for i in range(1)]     # create only one flow each state dimension,
            # need the copy.deepcopy so that each class has its own instance

        self.flow = initialize_flows(flow)

        # recognition network for initial state x_0
        self.recognet = LSTMRecognition(dim_outputs=self.output_dim,
                                        dim_inputs=self.input_dim,
                                        dim_states=self.state_dim,
                                        length=self.seq_len)

        # encode y_{1:T} into hidden states
        self.encodeNet = LSTMencoder(dim_outputs=self.output_dim,
                                    dim_inputs=self.input_dim,
                                    dim_states=self.state_dim,
                                    length=self.seq_len,
                                    hidden_size=hidden_size,
                                    num_layers=2,
                                    batch_first=True,
                                    bd=True)

        # variational distribution q(x_{1:T})
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

        # change 'cov_g' into a tensor
        cov_g = torch.tensor(cov_g, dtype=dtype).to(device)

        ''' ---------------  1.  KL[ q(x0) || p(x0) ]  --------------- '''
        # qx0: shape: batch_size x dim_state
        qx0 = self.recognet(obsr, input_sequence)

        # construct px0
        px0_mean = torch.zeros_like(qx0.mean)                    # shape: batch_size x dim_state
        px0_cov = torch.diag_embed(torch.ones_like(qx0.mean))    # shape: batch_size x dim_state

        px0 = MultivariateNormal(px0_mean, px0_cov)
        qm0_KL = KL_divergence(qx0, px0).mean()                  # take average over batch_size

        '''---------------  2.  encode y_{1:T} using LSTM   --------------- '''
        hidden_all = self.encodeNet(output_sequence=obsr, input_sequence=input_sequence)  # shape: batch_size x seq_len x (1+bd)*hidden_size

        '''---------------  initialization   ---------------'''
        result_gp_dynamic = torch.tensor(0.).to(device)
        data_fit = torch.tensor(0.).to(device)
        const = self.state_dim
        Hx = torch.tensor(0.).to(device) + const * 0.5 + 0.5 * const * math.log(2 * math.pi)

        x_t_1 = qx0.rsample()       # shape: batch_size x dim_state
        for t in range(self.seq_len):
            hidden_t = hidden_all[:, t]
            x_t, qxt = self.postNet(x_t_1=x_t_1, hidden=hidden_t)        # shape: batch_size x dim_states
            sigma_t = qxt.covariance_matrix

            Hx_tmp =  0.5 * torch.logdet(sigma_t).mean()                 # average over batch_size
            Hx = Hx + Hx_tmp.div(self.seq_len)                           # calculate the entropy term

            '''---------------    data-fit (log-likelihood term)  based on y_t      ---------------'''
            yt = obsr[:, t]                                             # shape: batch_size x dim_output
            # observation covariance matrix
            obser_covar = cov_g * torch.eye(self.output_dim).expand(batch_size, self.output_dim, self.output_dim).to(device)

            # mean vector of the observable latent states
            mu_t = x_t[:, emi_idx]
            # covariance matrix of the observable latent states
            V_t = sigma_t[:, emi_idx]
            V_t = V_t[:, :, emi_idx]
            V_t = V_t.view(batch_size, self.output_dim, self.output_dim)

            # E_{q(x_t)} [ log p(y_t | x_t) ]
            y_dist = MultivariateNormal(mu_t, obser_covar)
            data_fit_tmp = y_dist.log_prob(yt).view(-1) - 0.5 * torch.diagonal(obser_covar.inverse().matmul(V_t),
                                                                               dim1=-1,
                                                                               dim2=-2).sum(dim=1)
            data_fit_tmp = data_fit_tmp.div(self.seq_len)
            data_fit = data_fit + data_fit_tmp.mean()

            ''' ---------------      TGP dynamics term           ---------------'''
            # x_t_1 shape: batch_size x dim_state.     GP dynamics, shape: dim_state x batch_size
            gp_input = x_t_1      # shape:  batch_size x dim_state
            if input_sequence is not None:
                c_t = input_sequence[:, t]                  # shape: batch_size x dim_input
                gp_input = torch.cat((c_t, x_t_1), dim=-1)  # shape: batch_size x (dim_input + dim_state)

            # TODO: check if '_result_gp_dynamic' has been divided by 'seq_len'
            # ELL: 4th argument: [dim_state, MB, dim_input+dim_state], 5th argument: [dim_state, MB]
            _result_gp_dynamic = ELL(self.transition,
                                     self.flow,
                                     self.likelihood,
                                     gp_input.repeat(self.state_dim, 1, 1),
                                     x_t.transpose(0,1))     # should output a scalar

            result_gp_dynamic = result_gp_dynamic + _result_gp_dynamic.div(self.seq_len)

            # update x_t_1 for next round
            x_t_1 = x_t

        '''---------------  Calculate the KL divergence term of variational GP  ---------------'''
        KL_div = self.transition.kl_divergence().div(self.seq_len)
        ELBO = -qm0_KL + Hx - KL_div + data_fit + result_gp_dynamic

        print()
        print(f" Entropy term: {Hx.item()}")
        print(f" x0 KL: {qm0_KL.item()} ")
        print(f" KL term: {KL_div} ")
        print(f" data-fit term: {data_fit.item()}")
        print(f" GP dynamic term: {result_gp_dynamic.item()}")
        print(f"ELBO: {ELBO.item()}")
        print()

        '''------------------------    update beta    ------------------------'''
        self.Rtmp  = (1 - self.alpha) * self.Rtmp + self.alpha * (data_fit.detach())
        self.beta = self.beta * math.exp(self.nu * (self.R0 - self.Rtmp))
        print(f"beta: {self.beta}")
        print()

        return -qm0_KL + Hx + (self.beta + 1) * (data_fit) + result_gp_dynamic - KL_div

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