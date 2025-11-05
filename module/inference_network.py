"""
inference network is using to learn the parameters of the variational distributions of latent state
e.g.,
 -- Joint Gaussian case: learn the [mt, Lt] in distribution q(xt | xt-1 ) = N(mt, Lt * Lt.transpose), t = 1,2,3, ..., T
"""

import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import TriangularLazyTensor, CholLazyTensor
from .utils import inverse_softplus, safe_softplus

class Inference(nn.Module):
    """Base Class for recognition Module.

    Parameters
    ----------
    dim_outputs: int.
        Dimension of the outputs.
    dim_inputs: int.
        Dimension of the inputs.
    dim_states: int.
        Dimension of the state.
    length: int.
        Inference length.
    """

    def __init__(self, dim_outputs: int, dim_inputs: int, dim_states: int, length: int ) -> None:
        super().__init__()
        self.dim_outputs = dim_outputs
        self.dim_inputs = dim_inputs
        self.dim_states = dim_states
        self.length = length


class LSTMencoder(Inference):
    """LSTM Based Inference Network. Need to use with PostNet

    With Markov Gaussian Structure. Based on the paper:
        Rahul G. Krishnan, et al.
            "Structured Inference Networks for Nonlinear State Space Models." AAAI 2017

    """

    def __init__(self, dim_outputs, dim_inputs, dim_states, length,
                 hidden_size=32, num_layers=2, batch_first=True, bd=True):
        super().__init__(dim_outputs, dim_inputs, dim_states, length)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bd = bd

        self.lstm = nn.LSTM(input_size=self.dim_inputs + self.dim_outputs,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=batch_first,
                            bidirectional=bd
                            )

    def forward(self, output_sequence, input_sequence=None):
        """Forward execution of the recognition model."""

        device = output_sequence.device

        if input_sequence is None:
            input_sequence = torch.tensor([]).to(device)
        else:
            input_sequence = input_sequence.flip(dims=[1])

        # Reshape input/output sequence:
        # output_sequence: [batch_sz x seq_len x dim_output]
        # input_sequence: [batch_sz x seq_len x dim_input]

        batch_size = output_sequence.shape[0]

        io_sequence = torch.cat((output_sequence, input_sequence.to(device)), dim=-1)

        num_layers = self.lstm.num_layers * (1 + self.lstm.bidirectional)

        hidden = (torch.zeros(num_layers, batch_size, self.lstm.hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, self.lstm.hidden_size).to(device))

        # out shpae: [batch_size, seq_len, (1+bd)*hidden_size], where we set 'batch_first = True' in LSTM
        out, _ = self.lstm(io_sequence, hidden)

        return out

class PostNet(nn.Module):
    """
    Parameterizes `q(x_t|x_{t-1}, y_{1:T})`, which is the basic building block of the inference (i.e. the variational distribution).
    The dependence on `y_{1:T}` is through the hidden state of the RNN

    With Markov Gaussian Structure. Based on the Structured Inference Networks mentioned in Section IV-B
    """
    def __init__(self, x_dim, h_dim, bd=True, mf_flag=True):
        super(PostNet, self).__init__()
        self.bd = bd
        self.flag_mf = mf_flag

        if not mf_flag:
            self.x_to_h = nn.Sequential( nn.Linear(x_dim, h_dim), nn.Tanh() )

        ''' 
        future work: 
        build the correlations between each dimensions of the latent state, i.e., out_feature = dim_states x dim_states
        '''
        self.h_to_inv_softplus_var = nn.Linear(in_features=h_dim, out_features=x_dim)
        self.h_to_mu = nn.Linear(in_features=h_dim, out_features=x_dim)

    def forward(self, x_t_1, hidden):
        """
        Given the latent x_t_1 at a particular time step t-1 as well as the hidden
        state of the RNN `h(y_{1:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(x_t|x_{t-1}, y_{1:T})`

        # hidden shape: [batch_size x  (1+bd) * hidden_size]
        # x_t_1 shape:  [batch_size x state_dim]
        """
        h_tmp = hidden.view(hidden.shape[0], 1+self.bd, -1)    # hidden shape: [batch_size x (1+bd) x hidden_size]
        h_combined = h_tmp.sum(1).div(2)                       # h_combined shape: [batch_size x  hidden_size]

        if not self.flag_mf:
            h_combined = 1/3 * (self.x_to_h(x_t_1) + 2 * h_combined) # combine the LSTM hidden state with a transformed version of x_t_1

        mu = self.h_to_mu(h_combined)
        inv_soft_plus_var = self.h_to_inv_softplus_var(h_combined)
        var = safe_softplus(inv_soft_plus_var)

        epsilon = torch.randn(x_t_1.size(), device=x_t_1.device) # sampling x by re-parameterization
        x_t = epsilon * torch.sqrt(var) + mu                     # [batch_size x dim_states]
        return x_t, MultivariateNormal(mu, torch.diag_embed(var)).add_jitter()


class MFInference(Inference):
    """LSTM Based Inference Network. Joint Gaussian variation distribution for the latent states.

    With Markov Gaussian Structure. Based on the paper:
        Eleftheriadis, Stefanos, et al.
            "Identification of Gaussian process state space models."
            Advances in neural information processing systems 30 (2017).

    """

    def __init__(self, dim_outputs, dim_inputs, dim_states, length,
                 hidden_size=32, num_layers=2, batch_first=True, bd=True):

        super().__init__(dim_outputs, dim_inputs, dim_states, length)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.dim_inputs + self.dim_outputs,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=batch_first,
                            bidirectional=bd
                            )

        in_features = self.hidden_size * (1+bd)
        self.At = nn.Linear(in_features=in_features, out_features=self.dim_states * self.dim_states)
        self.raw_covar = nn.Linear(in_features=in_features, out_features=self.dim_states * self.dim_states)

    def forward(self, output_sequence, input_sequence=None):
        """Forward execution of the recognition model."""

        device = self.raw_covar.bias.device

        if input_sequence is None:
            input_sequence = torch.tensor([]).to(device)
        else:
            input_sequence = input_sequence.flip(dims=[1])

        # Reshape input/output sequence:
        # output_sequence: [batch_sz x seq_len x dim_output]
        # input_sequence: [batch_sz x seq_len x dim_input]

        batch_size = output_sequence.shape[0]

        io_sequence = torch.cat((output_sequence, input_sequence), dim=-1)

        num_layers = self.lstm.num_layers * (1 + self.lstm.bidirectional)

        hidden = (torch.zeros(num_layers, batch_size, self.lstm.hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, self.lstm.hidden_size).to(device))

        # out shpae: [batch_size, seq_len, (1+bd)*hidden_size], where we set 'batch_first = True' in LSTM
        out, _ = self.lstm(io_sequence, hidden)

        x = out

        At_all = self.At(x)  # At_all shape: batch_size x seq_len x (dim_state x dim_state)
        At_all = At_all.reshape(batch_size, self.length, self.dim_states, self.dim_states)

        raw_cov = self.raw_covar(x) # raw_cov shape: batch_size x seq_len x (dim_state*dim_state)
        raw_cov = raw_cov.reshape(batch_size, self.length, self.dim_states, self.dim_states)
        raw_cov = safe_softplus(raw_cov)

        # First make the cholesky factor is lower triangular
        lower_mask = torch.ones(raw_cov.shape[-2:]).tril(0).expand(batch_size, self.length, self.dim_states, self.dim_states).to(device)
        Lt_all = raw_cov.mul(lower_mask)

        return At_all, Lt_all

class NMFPostNet(nn.Module):
    """
    Parameterizes `q(x_t|f_{t}, y_{1:T})`, which is the basic building block of the inference (i.e. the variational distribution).
    The dependence on `y_{1:T}` is through the hidden state of the RNN

    With structured inference network.
    """
    def __init__(self, f_dim, h_dim, bd=True, mf_flag=True):
        super(NMFPostNet, self).__init__()
        self.bd = bd
        self.flag_mf = mf_flag


        self.f_to_h = nn.Sequential( nn.Linear(f_dim, h_dim), nn.Tanh() )

        ''' 
        future work: 
        build the correlations between each dimensions of the latent state, i.e., out_feature = dim_states x dim_states
        '''
        self.h_to_inv_softplus_var = nn.Linear(in_features=h_dim, out_features=f_dim)
        self.h_to_mu = nn.Linear(in_features=h_dim, out_features=f_dim)

    def forward(self, f_t, hidden):
        """
        Given the latent transition function value f_t at a particular time step t as well as the hidden
        state of the RNN `h(y_{1:T})` we return the mean and scale vectors that parameterize the (diagonal)
        Gaussian distribution `q(x_t|f_{t}, y_{1:T})`

        # hidden shape: [batch_size x  (1+bd) * hidden_size]
        # f_t shape:  [N_MC x batch_size x state_dim]

        return: 1. x_t ~ q(x_t|f_{t}, y_{1:T});
                2. q(x_t|f_{t}, y_{1:T});
        """
        h_tmp = hidden.view(hidden.shape[0], 1+self.bd, -1)    # hidden shape: [batch_size x (1+bd) x hidden_size]
        h_combined = h_tmp.sum(1).div(2)                       # h_combined shape: [batch_size x  hidden_size]

        # shape: N_MC x batch_size x hidden_size
        h_combined = 1/3 * (self.f_to_h(f_t) + 2 * h_combined) # combine the LSTM hidden state with a transformed version of f_t

        mu = self.h_to_mu(h_combined)  #shape: N_MC x batch_size x state_dim
        inv_soft_plus_var = self.h_to_inv_softplus_var(h_combined)
        var = safe_softplus(inv_soft_plus_var)
        #
        # logvar = self.h_to_inv_softplus_var(h_combined)
        # var = torch.exp(logvar)

        epsilon = torch.randn(f_t.size(), device=f_t.device) # sampling x by re-parameterization
        x_t = epsilon * torch.sqrt(var) + mu                 # [N_MC x batch_size x state_dim]
        return x_t, MultivariateNormal(mu, torch.diag_embed(var))