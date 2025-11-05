import numpy as np
import torch
import math
import platform

def Kink_function(trajectory_length = 2000, state_int = 0.5, process_noise_sd = np.sqrt(0.01),
                  observation_noise_sd = np.sqrt(0.1)):
    '''
    Kink function data generation
    '''
    states, observations = np.zeros(trajectory_length), np.zeros(trajectory_length)
    states[0], observations[0] = state_int, state_int + np.random.normal(0.0, observation_noise_sd)
    for i in range(trajectory_length-1):
        f = 0.8 + (states[i] + 0.2) * (1 - 5 / (1 + np.exp(-2 * states[i])))
        states[i+1] = f + np.random.normal(0.0, process_noise_sd)
        observations[i+1] = states[i + 1] + np.random.normal(0.0, observation_noise_sd)
    return states, observations

def KS_function(trajectory_length = 2000, state_int = 0.5, process_noise_sd = np.sqrt(0.01),
                  observation_noise_sd = np.sqrt(0.1)):
    '''
    Kink+Step function data generation
    '''
    states, observations = np.zeros(trajectory_length), np.zeros(trajectory_length)
    states[0], observations[0] = state_int, state_int + np.random.normal(0.0, observation_noise_sd)
    for i in range(trajectory_length-1):
        if ((states[i]<5 and states[i]>=4) or (states[i]<3) ):
            f = states[i] + 1
        elif (states[i]<4 and states[i]>=3):
            f = 0.0
        else:
            f = 16 - 2 * states[i]
        states[i+1] = f + np.random.normal(0.0, process_noise_sd)
        observations[i+1] = states[i + 1] + np.random.normal(0.0, observation_noise_sd)
    return states, observations

def xsin(trajectory_length = 500, observation_noise_sd = np.sqrt(0.1)):
    '''
    y = x * sin(4 * pi * x)
    '''
    x = np.linspace(0, 8, trajectory_length)
    f = x * np.sin(1 * np.pi * x)
    y = f + np.random.normal(loc=0.0, scale=observation_noise_sd, size=f.shape)
    return x, y


def move_one(A):
    b = np.zeros(len(A))
    for i in range(len(A) - 1):
        b[i] = (A[i + 1])
    return b

# def get_minibatch(obsv, batch_size, device='cpu'):
#     """
#     get minibatch data
#     Input:
#         obsv: observed time series. Shape: [episodes, seq_len, obser_dim]
#         batch_size: batch size
#     Output: Tensor
#         shape: [ batch_size, seq_len, obser_dim]
#     """
#     indices = torch.randperm(obsv.shape[0])[:batch_size]
#     return obsv[indices, :,  :].to(device), indices


def parse_inducing_points(dim_inputs: int = 1, dim_outputs: int = 1, number_points: int = 20,
                          strategy: str = 'linspace', scale: float = 3):
    """Initialize inducing points for variational GP.
    Parameters
    ----------
    dim_inputs: int.
        Input dimensionality.
    number_points: int.
        Number of inducing points.
    strategy: str, optional.
        Strategy to generate inducing points (by default normal).
        Either either normal, uniform or linspace.
    scale: float, optional.
        Scale of inducing points (default 1)

    Returns
    -------
    inducing_point: torch.Tensor.
        Inducing points with shape [dim_outputs x num_inducing_points x dim_inputs]
    learn_loc: bool.
        Flag that indicates if inducing points are learnable.

    Examples
    --------
    >>> num_points, dim_inputs, dim_outputs = 24, 8, 4
    >>> for strategy in ['normal', 'uniform', 'linspace']:
    ...     ip, l = parse_inducing_points(dim_inputs, dim_outputs, num_points,
    ...     strategy, 2.)
    ...     assert type(ip) == torch.Tensor
    ...     assert ip.shape == torch.Size([dim_outputs, num_points, dim_inputs])
    ...     assert l
    """
    if strategy == 'normal':
        ip = scale * torch.randn((dim_outputs, number_points, dim_inputs))
    elif strategy == 'uniform':
        ip = scale * torch.rand((dim_outputs, number_points, dim_inputs))
    elif strategy == 'linspace':
        lin_points = int(np.ceil(number_points ** (1 / dim_inputs)))
        ip = np.linspace(-scale, scale * 0.8, lin_points)
        ip = np.array(np.meshgrid(*([ip] * dim_inputs))).reshape(dim_inputs, -1).T
        idx = np.random.choice(np.arange(ip.shape[0]), size=number_points, replace=False)
        ip = torch.from_numpy(ip[idx]).float().repeat(dim_outputs, 1, 1)
    else:
        raise NotImplementedError("inducing point {} not implemented.".format(strategy))
    assert ip.shape == torch.Size([dim_outputs, number_points, dim_inputs])
    return ip


def reset_seed(seed: int) -> None:
    # 生成随机数，以便固定后续随机数，方便复现代码
    np.random.seed(seed)
    # 没有使用GPU的时候设置的固定生成的随机数
    np.random.seed(seed)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(seed)
    # torch.cuda.manual_seed()为当前GPU设置随机种子
    torch.cuda.manual_seed(seed)

## Config Variables
# torch_version = '1.8.1'
device='cuda:1'
# device = 'cpu'
dtype = torch.float
torch.set_default_dtype(dtype)
is_linux = 'linux' in platform.platform().lower()
reset_seed(seed=0)

## Constant definitions
pi = torch.tensor(math.pi).to(device)

## Computation constants
quad_points     = 100 # number of quadrature points used in integrations
constant_jitter = None # if provided, then this jitter value is added always when computing cholesky factors
global_jitter   = None # if None, then it uses 1e-8 with float 64 and 1-6 with float 32 precission when a cholesky error occurs
