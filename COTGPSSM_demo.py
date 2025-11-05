# main function of running COTGPSSM
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from module.TGPSSMs import COTGPSSM_E as TGPSSM
from src import utils as cg
from src.utils_plot import plot_1D_results # plot_tgp

cg.reset_seed(0)
device = cg.device

number_ips = 15
num_epoch = 3000
# func ='kinkfunc'
func ='ksfunc'

process_noise_sd = 0.05
observation_noise_sd = 0.2

save_fig = True
save_model = True

# data generation (generate 200 episodes of length 11 to train GPSSM)
if func == 'kinkfunc':
    lo = -3.15
    up = 1.15
    # hidden_size = 32
    state_dim = 1
    output_dim = 1
    input_dim = 0
    episode = 30
    seq_len = 20
    batch_size = episode # full batch training

    true_state_np, obsers_np = cg.Kink_function(trajectory_length = episode * seq_len * state_dim,
                                                state_int = 0.5,
                                                process_noise_sd = process_noise_sd,
                                                observation_noise_sd = observation_noise_sd)

    # initialize inducing points: shape: state_dim x number_ips x state_dim
    inducing_points = torch.linspace(lo, up, number_ips)
    inducing_points = inducing_points.repeat(state_dim, state_dim, 1).permute(0, 2, 1)
    # inducing_points = cg.parse_inducing_points(dim_inputs=state_dim, dim_outputs=state_dim, number_points=number_ips,
    #                                            strategy='linspace', scale=3)

    # 画出数据
    fig, (ax1, ax) = plt.subplots(1, 2, figsize=(10, 5))
    ax.plot(obsers_np, cg.move_one(obsers_np), 'r*', label='Data', markersize=10)
    ax.set(xlabel="y[t]", ylabel="y[t+1]")
    ax1.plot(true_state_np, cg.move_one(true_state_np), 'b*', label='Data', markersize=10)
    ax1.set(xlabel="x[t]", ylabel="x[t+1]")
    plt.show()

elif func == 'ksfunc':
    lo = -0.5
    up = 6.5
    # hidden_size = 32
    state_dim = 1
    output_dim = 1
    input_dim = 0
    episode = 30
    seq_len = 20
    batch_size = episode # full batch training

    true_state_np, obsers_np = cg.KS_function(trajectory_length = episode * seq_len * state_dim,
                                              state_int = 0.5,
                                              process_noise_sd = process_noise_sd,
                                              observation_noise_sd = observation_noise_sd)

    # initialize inducing points: shape: state_dim x number_ips x state_dim
    inducing_points = torch.linspace(lo, up, number_ips)
    inducing_points = inducing_points.repeat(state_dim, state_dim, 1).permute(0, 2, 1)
    # inducing_points = cg.parse_inducing_points(dim_inputs=state_dim, dim_outputs=state_dim, number_points=number_ips,
    #                                            strategy='uniform', scale=6)

    # 画出数据
    fig, (ax1, ax) = plt.subplots(1, 2, figsize=(10, 5))
    ax.plot(obsers_np, cg.move_one(obsers_np), 'r*', label='Data', markersize=10)
    ax.set(xlabel="y[t]", ylabel="y[t+1]")
    ax1.plot(true_state_np, cg.move_one(true_state_np), 'b*', label='Data', markersize=10)
    ax1.set(xlabel="x[t]", ylabel="x[t+1]")
    plt.show()
else:
    raise NotImplementedError("Function format not implemented.")


#准备数据
obsers = torch.tensor(obsers_np.reshape([episode, seq_len, output_dim]), dtype=torch.float).to(device)
true_state = torch.tensor(true_state_np.reshape([episode, seq_len, state_dim]), dtype=torch.float).to(device)

# 准备模型
num_flow_blocks = 2
model = TGPSSM(state_dim=state_dim, output_dim=output_dim, seq_len=seq_len, inducing_points=inducing_points,
               input_dim=input_dim, process_noise_sd=process_noise_sd,
               num_flow_blocks=num_flow_blocks, mf_flag=False, separate_flow=False).to(device)

lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dir = f'fig/fig_newFlow/{model._get_name()}/{func}_flBlock{num_flow_blocks}/obser_noise_std_{observation_noise_sd}/'
if not os.path.exists(dir):
    os.makedirs(dir)

# computing the initial R0
iniNumIter = 200
R1 = 0
for t in range(iniNumIter):
    model.train()
    optimizer.zero_grad()
    RO, R1 = model.calculate_r0(obsers, observation_noise_sd ** 2)  # for kink function and KS func
    loss = -RO
    loss.backward()
    optimizer.step()
    print(f"iteration: {t},     R0: {RO}")

# initialize the quality of R
model.R0 = 0.9 * R1.detach().cpu().numpy()

log_dir = dir + f"_epoch14990.pt"
if os.path.exists(log_dir):
    checkpoint = torch.load(log_dir)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    print('Load epoch {} successfully!'.format(start_epoch))
else:
    start_epoch = 0
    losses = []
    print('No existing models, training from beginning!')

# Record the training time
start_time = time.time()
losses = []
epochiter = tqdm(range(start_epoch, start_epoch+num_epoch), desc='Epoch:')
for epoch in epochiter:
    model.train()
    optimizer.zero_grad()
    # ELBO = model(obsers, observation_noise_sd ** 2, emi_idx=[0,1,2])
    ELBO = model(obsers, observation_noise_sd**2)
    loss = -ELBO
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    epochiter.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})

    if epoch % 200 == 0: # plot the results
        MSE_preTGP = plot_1D_results(model=model, epoch=epoch, func=func, save=save_fig, path=dir)

    if epoch%500==0:
        ''' save model '''
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch,
                 'losses': losses}
        if save_model:
            log_dir = dir + f"_epoch{epoch}.pt"
            torch.save(state, log_dir)

end_time = time.time()

# 结果展示
model.eval()

plt.figure(figsize=(6, 6))
plt.plot(np.arange(len(losses)), -np.array(losses), c='r', label='ELBO (train)')
plt.xscale('log')
plt.title(r'training loss, {} data'.format(func), fontsize=15)
plt.ylabel(r'$\mathcal{L}$', fontsize=15)
plt.legend(fontsize=12)
if save_fig:
    plt.savefig(dir+f'Training_loss_{func}.png', bbox_inches="tight")
else:
    plt.show()

MSE_preTGP = plot_1D_results(model=model, epoch=epoch, func=func, save=save_fig, path=dir)


''' save model '''
state = {'model': model.state_dict(),
         'optimizer': optimizer.state_dict(),
         'epoch': epoch,
         'losses':losses}
if save_model:
    log_dir = dir + f"_epoch{epoch}.pt"
    torch.save(state, log_dir)
