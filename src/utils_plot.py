import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
import gpytorch
from .utils_func import predictive_distribution, KS_func


def plot_1D_all(model, epoch, func='kinkfunc',save=False, path='./fig_MF1D/2layer_learned_kink_Epoch'):
    dtype = model.transition.variational_strategy.inducing_points.dtype
    device = model.transition.variational_strategy.inducing_points.device

    fontsize = 28
    N_test = 100

    if func == 'ksfunc':
        label = "Kink-step function"
        X_test = np.linspace(-0.5, 6.5, N_test)
        y_test = np.zeros(N_test)
        for i in range(N_test):
            y_test[i] = KS_func(X_test[i])

    elif func == 'kinkfunc':
        label = "Kink function"
        X_test = np.linspace(-3.15, 1.15, N_test)
        y_test = 0.8 + (X_test + 0.2) * (1 - 5 / (1 + np.exp(-2 * X_test)))

    else:
        raise NotImplementedError("The 'func' input only supports kinkfunc and KSfunc")

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_y = torch.tensor(y_test, dtype=torch.float).to(device)
        test_x = torch.tensor(X_test, dtype=torch.float).to(device)
        test_xx = test_x.reshape(-1, 1)
        # test_xx = test_x.reshape(-1,)

        dim_outputs, batch_size, dim_inputs = model.transition.variational_strategy.inducing_points.shape
        # test_xx = test_x.reshape(-1, 1)
        test_xx = test_x.reshape(dim_outputs, -1, dim_inputs)
        pred_val_mean, pred_val_variance, qf_mean, qf_variance = predictive_distribution(model.transition,
                                                                                         model.likelihood,
                                                                                         model.flow,
                                                                                         test_xx)

        MSE_preGP = 1 / (N_test) * np.linalg.norm(pred_val_mean.cpu().numpy().reshape(-1) - y_test) ** 2

        # inducing points
        U, _, _, _ = predictive_distribution(model.transition,
                                             model.likelihood,
                                             model.flow,
                                             model.transition.variational_strategy.inducing_points)

        lower, upper = pred_val_mean - 2 * torch.sqrt(pred_val_variance), \
                       pred_val_mean + 2 * torch.sqrt(pred_val_variance)

        lower, upper = lower.reshape(-1, ), upper.reshape(-1, )


    with torch.no_grad():

        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Plot training data as black starss
        ax.plot(model.transition.variational_strategy.inducing_points.cpu().numpy().reshape(-1, ),
                U.cpu().numpy().reshape(-1, ), 'g*', label='inducing points', markersize=10)

        # Plot test data as read stars
        ax.plot(X_test, y_test, 'r', label=label)
        # Plot predictive means as blue line
        ax.plot(X_test, pred_val_mean.cpu().numpy().reshape(-1, ), 'b', label='learned function')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.2, label='95% CI')
        ax.legend(loc=0, fontsize=fontsize)
        # plt.title(f"Epoch: {epoch}", fontsize=15)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()
        if func == 'ksfunc':
            ax.set_xlim([-0.5, 6.5])
        elif func == 'kinkfunc':
            ax.set_xlim([-3.15, 1.15])
        else:
            raise NotImplementedError("The 'func' input only supports kinkfunc and KSfunc")

        if save:
            plt.savefig(path + f"func_{func}_epoch_{epoch}.pdf")
        else:
            plt.show()

    return MSE_preGP





def plot_1D_results(model, epoch, func='kinkfunc',save=False, path='./fig_MF1D/2layer_learned_kink_Epoch'):
    """
    model:  GPSSM, JOGPSSM, COGPSSM, JOTGPSSM, COTGPSSM, JOTGPSSM_E, COTGPSSM_E
    epoch:
    func: kinkfunc, ksfunc
    save: True or False
    path: directional path
    """
    dtype = model.transition.variational_strategy.inducing_points.dtype
    device = model.transition.variational_strategy.inducing_points.device

    fontsize = 28
    N_test = 100

    if func == 'ksfunc':
        label = "Kink-step function"
        X_test = np.linspace(-0.5, 6.5, N_test)
        y_test = np.zeros(N_test)
        for i in range(N_test):
            y_test[i] = KS_func(X_test[i])

    elif func == 'kinkfunc':
        label = "Kink function"
        X_test = np.linspace(-3.15, 1.15, N_test)
        y_test = 0.8 + (X_test + 0.2) * (1 - 5 / (1 + np.exp(-2 * X_test)))

    else:
        raise NotImplementedError("The 'func' input only supports kinkfunc and KSfunc")

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_y = torch.tensor(y_test, dtype=torch.float).to(device)
        test_x = torch.tensor(X_test, dtype=torch.float).to(device)
        test_xx = test_x.reshape(-1, 1)
        # test_xx = test_x.reshape(-1,)

        if model._get_name() == 'GPSSM' or model._get_name() =='JOGPSSM' or model._get_name() =='COGPSSM':

            observed_pred = model.likelihood(model.transition(test_xx))
            pred_val_mean = observed_pred.mean

            U = model.transition(model.transition.variational_strategy.inducing_points).mean
            MSE_preGP = 1 / (N_test) * np.linalg.norm(pred_val_mean.cpu().numpy().reshape(-1) - y_test) ** 2

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            lower, upper = lower.reshape(-1, ), upper.reshape(-1, )

        elif model._get_name() == 'JOTGPSSM'or model._get_name() =='COTGPSSM':
            qf = model.transition(test_xx)

            # shape: N_MC x batch_size x state_dim
            ft = qf.rsample(torch.Size([model.N_MC])).transpose(-1, -2)
            f_k_t = model.flow(ft)
            observed_pred = model.likelihood(f_k_t)

            # shape: batch_size x state_dim
            pred_val_mean = observed_pred.mean.mean(dim=0)
            pred_val_var = observed_pred.mean.var(dim=0)

            U = model.transition(model.transition.variational_strategy.inducing_points).mean
            U = model.flow(U.transpose(0,1))

            MSE_preGP = 1 / (N_test) * np.linalg.norm(pred_val_mean.cpu().numpy().reshape(-1) - y_test) ** 2

            # Get upper and lower confidence bounds
            var = pred_val_var + model.likelihood.noise
            lower, upper = pred_val_mean - 2 * var.sqrt(), pred_val_mean + 2 * var.sqrt()
            lower, upper = lower.reshape(-1, ), upper.reshape(-1, )



        elif model._get_name() == 'JOTGPSSM_E'or model._get_name() =='COTGPSSM_E':
            dim_outputs, batch_size, dim_inputs = model.transition.variational_strategy.inducing_points.shape
            # test_xx = test_x.reshape(-1, 1)
            test_xx = test_x.reshape(dim_outputs, -1, dim_inputs)
            pred_val_mean, pred_val_variance, qf_mean, qf_variance = predictive_distribution(model.transition,
                                                                                             model.likelihood,
                                                                                             model.flow,
                                                                                             test_xx)

            MSE_preGP = 1 / (N_test) * np.linalg.norm(pred_val_mean.cpu().numpy().reshape(-1) - y_test) ** 2

            # inducing points
            U, _, _, _ = predictive_distribution(model.transition,
                                                 model.likelihood,
                                                 model.flow,
                                                 model.transition.variational_strategy.inducing_points)

            lower, upper = pred_val_mean - 2 * torch.sqrt(pred_val_variance), \
                           pred_val_mean + 2 * torch.sqrt(pred_val_variance)

            lower, upper = lower.reshape(-1, ), upper.reshape(-1, )

        else:
            raise NotImplementedError("The model type is not supported")

        with torch.no_grad():

            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(15, 10))

            # Plot training data as black starss
            ax.plot(model.transition.variational_strategy.inducing_points.cpu().numpy().reshape(-1, ),
                    U.cpu().numpy().reshape(-1, ), 'g*', label='inducing points',  markersize=10)

            # Plot test data as read stars
            ax.plot(X_test, y_test, 'r', label=label)
            # Plot predictive means as blue line
            ax.plot(X_test, pred_val_mean.cpu().numpy().reshape(-1, ), 'b', label='learned function')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.2, label='95% CI')
            ax.legend(loc=0, fontsize=fontsize)
            # plt.title(f"Epoch: {epoch}", fontsize=15)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.grid()
            if func == 'ksfunc':
                ax.set_xlim([-0.5, 6.5])
            elif func == 'kinkfunc':
                ax.set_xlim([-3.15, 1.15])
            else:
                raise NotImplementedError("The 'func' input only supports kinkfunc and KSfunc")

            if save:
                plt.savefig(path + f"func_{func}_epoch_{epoch}.pdf")
            else:
                plt.show()

    return MSE_preGP



