import torch
import math

def KS_func(x):
    if ((x < 5 and x >= 4) or (x < 3)):
        f = x + 1
    elif (x < 4 and x >= 3):
        f = 0
    else:
        f = 16 - 2 * x
    return f

def kink_func(x):
    f = 0.8 + (x + 0.2) * (1 - 5 / (1 + torch.exp(-2 * x)))
    return f

def xsin(x):
    f = x * torch.sin(1 * math.pi * x)
    return f


def predictive_distribution(model, likelihood, G_matrix, X: torch.tensor, diagonal: bool = True):
    """ This function computes the moments 1 and 2 from the predictive distribution.
        It also returns the posterior mean and covariance over latent functions.

        p(Y*|X*) = \int p(y*|G(f*)) q(f*,f|u) q(u) df*,df,du

            # Homoceodastic Gaussian observation model p(y|f)
            # GP variational distribution q(f)
            # G() represents a non-linear transformation

            Args:
                    `X`                (torch.tensor)  :->: input locations where the predictive is computed. Can have shape (MB,Dx) or (Dy,MB,Dx)
                    `diagonal`         (bool)          :->: if true, samples are drawn independently. For the moment is always true.
                    `S_MC_NNet`        (int)           :->: Number of samples from the dropout distribution is fully_bayesian is true

            Returns:
                    `m1`       (torch.tensor)  :->:  Predictive mean with shape (Dy,MB)
                    `m2`       (torch.tensor)  :->:  Predictive variance with shape (Dy,MB). Takes None for classification likelihoods
                    `mean_q_f` (torch.tensor)  :->:  Posterior mean of q(f) with shape (Dy,MB,1) [same shape as returned by marginal_variational_qf]
                    `cov_q_f`  (torch.tensor)  :->:  Posterior covariance of q(f) with shape (Dy,MB,1) [same shape as returned by marginal_variational_qf]

    """
    assert len(X.shape) == 3, "Bad input specificaton"
    model.eval()
    G_matrix.eval()
    likelihood.eval()  # set parameters for eval mode. Batch normalization, dropout etc

    with torch.no_grad():
        if not diagonal:
            raise NotImplemented("This function does not support returning the predictive distribution with correlations")

        qf = model(X)

        MOMENTS = likelihood.marginal_moments(qf.mean, qf.variance, diagonal=True, flow=G_matrix, X=X)
        # diagonal True always. Is an element only used by the sparse_MF_GP with SVI. Diag = False is used by standard GP's marginal likelihood

        m1, m2 = MOMENTS
    return m1, m2, qf.mean, qf.variance