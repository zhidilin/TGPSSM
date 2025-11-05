import sys
sys.path.append('../../')
import gpytorch
import torch
import torch.nn as nn
import torch.distributions as td
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
import src.utils as cg
# from gpytorch.variational import IndependentMultitaskVariationalStrategy


class IndependentMultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, dim_state, inducing_points):

        # task 数量等于 latent state 的数量
        self.state_dim = dim_state

        ### inducing_points = torch.rand(num_tasks, 16, 1)

        # Let's use same set of inducing points for each task
        # inducing_points shape: dim_state x num_ips x (dim_state + dim_input)
        assert(self.state_dim == inducing_points.shape[0])
        num_ips = inducing_points.size(-2)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_ips,
                                                                   batch_shape=torch.Size([self.state_dim])
                                                                   )

        variational_strategy = VariationalStrategy(self,
                                                   inducing_points=inducing_points,
                                                   variational_distribution=variational_distribution,
                                                   learn_inducing_locations=True
                                                   )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = ConstantMean(batch_shape=torch.Size([self.state_dim]))
        self.covar_module = ScaleKernel(MaternKernel(batch_shape=torch.Size([self.state_dim])),
                                        batch_shape=torch.Size([self.state_dim])
                                        )
        # self.covar_module = ScaleKernel(RBFKernel(batch_shape=torch.Size([self.state_dim])),
        #                                 batch_shape=torch.Size([self.state_dim])
        #                                 )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def kl_divergence(self):
        """Get the KL-Divergence of the Model."""
        return self.variational_strategy.kl_divergence().sum()

    # def post_f_condtion_u(self, x, U):
    #     '''
    #     # for non-mean-field cases
    #     U shape: N_MC x state_dim x num_induc,
    #     x shape: N_MC x batch_size x (state_dim + input_dim)
    #     '''
    #     state_dim = U.shape[1]
    #     assert len(x.shape) == 3, 'Bad input x, expected shape (N_MC x batch_size x (state_dim + input_dim) )'
    #     N_MC, batch_size, x_dim = x.shape
    #     # U = U.permute(1, 0, 2)    # shape: state_dim x N_MC x num_induc
    #     _, num_induc, _ = self.variational_strategy.inducing_points.shape
    #     inducing_points = self.variational_strategy.inducing_points
    #
    #     # shape [N_MC, state_dim, num_induc, (state_dim + input_dim)]
    #     inducing_points = inducing_points.expand(N_MC, state_dim, num_induc, x_dim)
    #
    #     # inducing_points = inducing_points.permute(1,0,2,3)   # shape [state_dim, N_MC, num_induc, (state_dim + input_dim)]
    #
    #     # x shape:  [N_MC, batch_x, (state_dim + input_dim)],  y shape [state_dim, batch_x, state_dim]
    #     x_input = x.repeat(state_dim, 1, 1, 1).permute(1,0,2,3)  # shape [N_MC, state_dim, batch_x, (state_dim + input_dim)]
    #
    #     full_input = torch.cat([inducing_points, x_input], dim=-2)
    #     full_output = self.forward(full_input)
    #     full_covar = full_output.lazy_covariance_matrix
    #
    #     # Covariance terms
    #     test_mean = full_output.mean[..., num_induc:]
    #     induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter().evaluate()
    #     induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
    #     data_data_covar = full_covar[..., num_induc:, num_induc:].add_jitter().evaluate()
    #
    #     # result term: tmp shape: N_MC x state_dim x batch_x x num_induc
    #     tmp = torch.matmul(induc_data_covar.permute(0, 1, 3, 2), torch.inverse(induc_induc_covar))
    #
    #     f_condition_u_cov = data_data_covar - torch.matmul(tmp, induc_data_covar) # shape: N_MC x state_dim x batch_size
    #     f_condition_u_mean = test_mean + torch.matmul(tmp, U.unsqueeze(dim=-1)).squeeze(dim=-1)
    #
    #     mu = f_condition_u_mean.permute(0,2,1)  # shape: N_MC x batch_size x state_dim
    #     var = f_condition_u_cov.diagonal(offset=0,dim1=-2,dim2=-1).permute(0,2,1)
    #
    #     return MultivariateNormal(mu, torch.diag_embed(var)).add_jitter()


# define a likelihood module
class GaussianNonLinearMean(nn.Module):
    """Place a GP over the mean of a Gaussian likelihood $p(y|G(f))$
    with noise variance $\sigma^2$ and with a NON linear transformation $G$ over $f$.
    It supports multi-output (independent) GPs and the possibility of sharing
    the noise between the different outputs. In this case integrations wrt to a
    Gaussian distribution can only be done with quadrature."""

    def __init__(self, out_dim: int, noise_init: float, noise_is_shared: bool, quadrature_points: int):
        super(GaussianNonLinearMean, self).__init__()

        self.out_dim = out_dim
        self.noise_is_shared = noise_is_shared

        if noise_is_shared:  # if noise is shared create one parameter and expand to out_dim shape
            log_var_noise = nn.Parameter(torch.ones(1, 1, dtype=cg.dtype) * torch.log(torch.tensor(noise_init, dtype=cg.dtype)))

        else:  # creates a vector of noise variance parameters.
            log_var_noise = nn.Parameter(torch.ones(out_dim, 1, dtype=cg.dtype) * torch.log(torch.tensor(noise_init, dtype=cg.dtype)))

        self.log_var_noise = log_var_noise

        self.quad_points = quadrature_points
        self.quadrature_distribution = GaussHermiteQuadrature1D(quadrature_points)

    ##  Log Batched Multivariate Gaussian: log N(x|mu,C) ##
    def batched_log_Gaussian(self, obs: torch.tensor, mean: torch.tensor, cov: torch.tensor, diagonal: bool, cov_is_inverse: bool) -> torch.tensor:
        """
        Computes a batched of * log p(obs|mean,cov) where p(y|f) is a  Gaussian distribution, with dimensionality N.
        Returns a vector of shape *.
        -0.5*N log 2pi -0.5*\log|Cov| -0.5[ obs^T Cov^{-1} obs -2 obs^TCov^{-1} mean + mean^TCov^{-1}mean]
                Args:
                        obs            :->: random variable with shape (*,N)
                        mean           :->: mean -> matrix of shape (*,N)
                        cov            :->: covariance -> Matrix of shape (*,N) if diagonal=True else batch of matrix (*,N,N)
                        diagonal       :->: if covariance is diagonal or not
                        cov_is_inverse :->: if the covariance provided is already the inverse

        #TODO: Check argument shapes
        """

        N = mean.size(-1)
        cte = N * torch.log(2 * cg.pi.to(cg.device).type(cg.dtype))

        if diagonal:
            log_det_C = torch.sum(torch.log(cov), -1)
            inv_C = cov
            if not cov_is_inverse:
                inv_C = 1. / cov  # Inversion given a diagonal matrix. Use torch.cholesky_solve for full matrix.
            else:
                log_det_C *= -1  # switch sign

            exp_arg = (obs * inv_C * obs).sum(-1) - 2 * (obs * inv_C * mean).sum(-1) + (mean * inv_C * mean).sum(-1)

        else:
            raise NotImplemented("log_Gaussian for full covariance matrix is not implemented yet.")
        return -0.5 * (cte + log_det_C + exp_arg)

    def log_non_linear(self, f: torch.tensor, Y: torch.tensor, noise_var: torch.tensor, flow: list, X: torch.tensor, **kwargs):
        """ Return the log likelihood of S Gaussian distributions, each of this S correspond to a quadrature point.
            The only samples f have to be warped with the composite flow G().
            -> f is assumed to be stacked samples of the same dimension of Y. Here we compute (apply lotus rule):

          \int \log p(y|fK) q(fK) dfK = \int \log p(y|fk) q(f0) df0 \approx 1/sqrt(pi) sum_i w_i { \log[ p( y | G( sqrt(2)\sigma f_i + mu), sigma^2 ) ] };

          where q(f0) is the initial distribution. We just face the problem of computing the expectation under a log Gaussian of a
          non-linear transformation of the mean, given by the flow.

                Args:
                        `f`         (torch.tensor)  :->:  Minibatched - latent function samples in (S,Dy,MB), being S the number of quadrature points and MB the minibatch.
                                                          This is directly given by the gpytorch.GaussHermiteQuadrature1D method in this format and corresponds to
                                                          \sqrt(2)\sigma f_i + mu see https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
                        `Y`         (torch.tensor)  :->:  Minibatched Observations in Dy x MB.
                        `noise_var` (torch.tensor)  :->:  Observation noise
                        'flow'      (CompositeFlow) :->:  Sequence of flows to be applied to each of the outputs
                        'X'         (torch.tensor)  :->:  Input locations used for input dependent flows. Has shape [Dy,S*MB,Dx] or shape [S*MB,Dx]. N
        """
        assert len(flow) == self.out_dim, "This likelihood only supports a flow per output_dim. Got {} for Dy {}".format(self.out_dim, len(flow))
        assert len(X.shape) == 3, 'Bad input X, expected (out_dim,MB*S,Dx)'
        assert X.size(0) == self.out_dim, 'Wrong first dimension in X, expected out_dim'

        S = self.quad_points
        MB = Y.size(1)
        Dy = self.out_dim

        Y = Y.view(Dy, MB, 1).repeat((S, 1, 1, 1))  # S,Dy,MB,1
        noise_var = noise_var.view(Dy, MB, 1).repeat((S, 1, 1, 1))  # S,Dy,MB,1

        fK = f.clone()

        # Be aware that as we expand X we will be performing self.quad_points forwards through the NNets for each output.
        # This might be inneficient unless pytorch only performs the operation once and returned the expanded dimension

        # expanded_size = [self.quad_points] + [-1]*(len(X.size()))
        # X = X.expand(expanded_size) # no need for this broadcasting as pytorch will broadcast automatically

        for idx, fl in enumerate(flow):
            # warp the samples
            fK[:, idx, :] = fl(f[:, idx, :], X[idx])

        fK = fK.view(S, Dy, MB, 1)
        # we add extra dimension so that batched_log_gaussian does not reduce minibatch dimension. This will be reduced at the end as the
        # GaussHermiteQudrature from gpytorch reduces S by default. Although sum is associative we prefer to separate for clarity.

        log_p_y = self.batched_log_Gaussian(obs=Y, mean=fK, cov=noise_var, diagonal=True, cov_is_inverse=False)  # (S,Dy,MB)

        return log_p_y  # return (S,Dy,MB) so that reduction is done for S.

    def expected_log_prob(self, Y, gauss_mean, gauss_cov, flow, X, **kwargs):
        """ Expected Log Likelihood

            Computes E_q(f) [\log p(y|G(f))] = \int q(f) \log p(y|G(f)) df \approx with quadrature

                - Acts on batched form. Hence returns vector of shape (Dy,)

            Args:

                `Y`             (torch.tensor)  :->:  Labels representing the mean. Shape (Dy,MB)
                `gauss_mean`    (torch.tensor)  :->:  mean from q(f). Shape (Dy,MB)
                `gauss_cov`     (torch.tensor)  :->:  diagonal covariance from q(f). Shape (Dy,MB)
                `non_linearity` (list)          :->:  List of callable representing the non linearity applied to the mean.
                `X`             (torch.tensor)  :->:  Input with shape (Dy,MB,Dx) or shape (MB,Dx). Needed for input dependent flows

        """

        assert len(flow) == self.out_dim, "The number of callables representing non linearities is different from out_dim"
        assert len(X.shape) == 3, 'Bad input X, expected (out_dim,MB*S,Dx)'
        assert X.size(0) == self.out_dim, 'Wrong first dimension in X, expected out_dim'

        if self.noise_is_shared:
            log_var_noise = self.log_var_noise.expand(self.out_dim, 1)
        else:
            log_var_noise = self.log_var_noise

        N = Y.size(1)
        C_y = torch.exp(log_var_noise).expand(-1, N)

        distr = td.Normal(gauss_mean, gauss_cov.sqrt())  # Distribution of shape (Dy,MB). Gpytorch samples from it

        log_likelihood_lambda = lambda f_samples: self.log_non_linear(f_samples, Y, C_y, flow, X)
        ELL = self.quadrature_distribution(log_likelihood_lambda, distr)
        # ELL shape is Dy x N

        return ELL

    def marginal_moments(self, gauss_mean, gauss_cov, flow, X, **kwargs):
        """ Computes the moments of order 1 and non centered 2 of the observation model integrated out w.r.t a Gaussian with means and covariances.
            There is a non linear relation between the mean and integrated variable

            p(y|x) = \int p(y|G(f)) p(f) df

            - Note that p(f) can only be diagonal as this function only supports quadrature integration.
            - Moment1: \widehat{\mu_y} = \frac{1}{\sqrt\pi} \sum^S_{s=1} w_s \mathtt{G}[\sqrt2 \sigma f_s + \mu]
            - Moment2: \sigma^2_o + \frac{1}{\sqrt\pi} \sum^S_{s=1} w_s \mathtt{G}[\sqrt2 \sigma f_s + \mu]^2 - \widehat{\mu_y}^2

            Args:
                `gauss_mean`    (torch.tensor) :->: mean from q(f). Shape (Dy,MB)
                `gauss_cov`     (torch.tensor) :->: covariance from q(f). Shape (Dy,MB) if diagonal is True, else (Dy,MB,MB). For the moment only supports diagonal true
                `non_linearity` (list)         :->: List of callable representing the non linearity applied to the mean.
                `X`             (torch.tensor) :->: Input locations used by input dependent flows
        """
        assert len(X.shape) == 3, 'Bad input X, expected (out_dim,MB*S,Dx)'
        assert X.size(0) == self.out_dim, 'Wrong first dimension in X, expected out_dim'

        if self.noise_is_shared:
            log_var_noise = self.log_var_noise.expand(self.out_dim, 1)
        else:
            log_var_noise = self.log_var_noise

        MB = gauss_mean.size(1)
        C_Y = torch.exp(log_var_noise).expand(-1, MB)  # shape (Dy,MB)

        # expanded_size = [self.quad_points] + [-1]*(len(X.size())-1)
        # X = X.expand(expanded_size) # no need for this expansion as pytorch will broadcast automatically
        def aux_moment1(f, _flow):
            # f.shape (S,Dy,MB)
            for idx, fl in enumerate(_flow):
                # warp the samples
                f[:, idx, :] = fl(f[:, idx, :], X[idx])
            return f

        def aux_moment2(f, _flow):
            # f.shape (S,Dy,MB)
            # x.shape (Dy,MB) # pytorch automatically broadcast to sum over S inside the flow fl
            for idx, fl in enumerate(_flow):
                # warp the samples
                f[:, idx, :] = fl(f[:, idx, :], X[idx])
            return f ** 2

        aux_moment1_lambda = lambda f_samples: aux_moment1(f_samples, flow)
        aux_moment2_lambda = lambda f_samples: aux_moment2(f_samples, flow)
        distr = td.Normal(gauss_mean, gauss_cov.sqrt())  # Distribution of shape (Dy,MB). Gpytorch samples from it

        m1 = self.quadrature_distribution(aux_moment1_lambda, distr)
        E_square_y = self.quadrature_distribution(aux_moment2_lambda, distr)
        m2 = C_Y + E_square_y - m1 ** 2

        return m1, m2

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


def ELBO(GP_model, G_matrix, likelihood, X, y, num_data, beta=1):
    """ Define the loss object: Evidence Lower Bound
     Args:
            GP_model: Variational sparse GP module
            G_matrix: flows module
            Likelihood: Likelihood module
            `X` (torch.tensor)  :->:  Inputs, shape: [dy, MB, dx]
            `y` (torch.tensor)  :->:  Targets, shape: [dy, MB]

                ELBO = \int log p(y|f) q(f|u) q(u) df,du -KLD[q||p]

                Returns possitive loss, i.e: ELBO = ELL - KLD; ELL and KLD

            """
    if len(X.shape) == 2:
        # repeat here as if not this operation will be done twice by the marginal_qf_parameter and likelihood to
        # work batched and multioutput respectively
        X = X.repeat(y.shape[-1], 1, 1)
    assert len(X.shape) == 3, 'Invalid input X.shape'

    qf = GP_model(X)  # output is a Gaussian distribution

    # 2nd argument: mean from q(f). Shape (Dy,MB)  #3rd argument: diagonal covariance from q(f). Shape (Dy,MB)
    ELL = likelihood.expected_log_prob(y, qf.mean, qf.variance, flow=G_matrix, X=X)
    ELL = ELL.sum(-1).div(y.shape[1])

    kl_divergence = GP_model.variational_strategy.kl_divergence().div(num_data / beta)
    ELBO = ELL - kl_divergence

    return ELBO


def ELL(GP_model, G_matrix, likelihood, X, y):
    """ Define the loss object: Evidence Lower Bound
         Args:
                GP_model: Variational sparse GP module
                G_matrix: flows module
                Likelihood: Likelihood module
                `X` (torch.tensor)  :->:  Inputs, shape: [dy, MB, dx]
                `y` (torch.tensor)  :->:  Targets, shape: [dy, MB]

                    ELL = \int log p(y|f) q(f|u) q(u) df,du

                    Returns possitive ELL

                """
    if len(X.shape) == 2:
        # repeat here as if not this operation will be done twice by the marginal_qf_parameter and likelihood to
        # work batched and multi-output respectively
        X = X.repeat(y.shape[-1], 1, 1)
    assert len(X.shape) == 3, 'Invalid input X.shape'

    qf = GP_model(X)  # output is a Gaussian distribution

    # 2nd argument: mean from q(f). Shape (Dy,MB)
    # #3rd argument: diagonal covariance from q(f). Shape (Dy,MB)
    ELL = likelihood.expected_log_prob(y, qf.mean, qf.variance, flow=G_matrix, X=X)
    ELL = ELL.sum(-1).div(y.shape[1])

    return ELL.mean()



# class GPModel(ApproximateGP):
#     def __init__(self, inducing_points, state_dim=1):
#         self.state_dim = state_dim
#         # inducing points shade: [number_inducing_points x input_dim]
#         variational_distribution = CholeskyVariationalDistribution(inducing_points.shape[0])
#         variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
#         super(GPModel, self).__init__(variational_strategy)
#         self.mean_module = gpytorch.means.LinearMean(input_size=self.state_dim)
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#
#     def __call__(self, state_input, **kwargs):
#         """Override call method to expand test inputs and not train inputs."""
#         batch_size, input_dim = state_input.shape
#         state_input = state_input.expand(self.state_dim, batch_size, input_dim).permute(1, 0, 2)
#
#         return ApproximateGP.__call__(self, state_input, **kwargs)
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#
#     def kl_divergence(self):
#         """Get the KL-Divergence of the Model."""
#         return self.variational_strategy.kl_divergence()