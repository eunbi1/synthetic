"""Various sampling methods."""

import torch
import numpy as np
from scipy import integrate
from scipy.special import gamma
from torchlevy import LevyStable
from scipy.special import jv, gamma, factorial
from torchquad import Simpson, MonteCarlo

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))

levy = LevyStable()

def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


def get_likelihood_fn( sde, device, hutchinson_type='Rademacher',noise_mode= 'isotropic',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5, alpha=1.8):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.
    Args:
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        inverse_scaler: The inverse data normalizer.
        hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
        rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
        atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
        method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.
        eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.
    Returns:
        A function that a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """

    def gamma_func(x):
        return torch.tensor(gamma(x)).to(device)

    def drift_fn(score_model, x, t):
        """The drift function of the reverse-time SDE."""
        # score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
        # # Probability flow ODE is a special case of Reverse SDE
        # rsde = sde.reverse(score_fn, probability_flow=True)
        # return rsde.sde(x, t)[0]
        if score_model.dimension ==1:
            x = x.reshape(-1,1)
            t = t.reshape(-1,1)

        score = score_model(x, t) * torch.pow(sde.marginal_std(t), -(sde.alpha-1))
        x_coeff = - sde.beta(t) / sde.alpha
        if alpha == 2:
                score_coeff = - sde.beta(t) / 2
        else:
                score_coeff = - sde.beta(t)
        # else :
        #     score = score_model(x, t) * torch.pow(sde.marginal_std(t), -(sde.alpha - 1))
        #     x_coeff = - (sde.beta(t) / sde.alpha)
        #     if alpha == 2:
        #         score_coeff = - (sde.beta(t) / 2)
        #     else:
        #         score_coeff = - (sde.beta(t))

        # print('x_coeff', x_coeff.shape)
        # print('x', x.shape)
        # print('score coe', score_coeff.shape)
        # print('score', score.shape)
        drift = x_coeff * x + score_coeff * score
        # print('drift', drift.shape )

        # if sde.alpha != 2.0:
        #     drift = torch.clamp(drift, -2.3, 2.3)

        return drift

    def div_fn(model, x, t, noise):
        return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)



    def likelihood_fn(score_model, data, noise_mode= noise_mode,inverse_scalar=0, t_0=1e-5, T=1,alpha=alpha):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.
        Args:
        model: A score model.
        data: A PyTorch tensor.
        Returns:
        bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
        z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
            probability flow ODE.
        nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        with torch.no_grad():
            shape = data.shape
        if hutchinson_type == 'Gaussian':
            epsilon = torch.randn_like(data)
        elif hutchinson_type == 'Rademacher':
            epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
        else:
            raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")


        def prior_log(x, d=2, alpha=alpha):

            dimension = x.shape[1]
            if alpha==2:
                log0 = -np.log(2*np.pi)/2
                return (-torch.abs(x) ** alpha/2 + log0)
            elif noise_mode == 'isotropic':
                if dimension==1 :
                    return torch.log(levy.pdf(x, alpha))
                else:
                    norm_x = torch.norm(x, dim=1)
                    d = x.shape[1]
                    def a(r_theta):
                        def bessel(r):
                            return torch.tensor(jv(d / 2 - 1, (r * norm_x).cpu().numpy()))

                        r = r_theta[:, None]
                        ret = r ** (d / 2) * bessel(r) * torch.exp(-r ** (alpha))
                        return ret

                    simp = Simpson()
                    intg_a = simp.integrate(a, dim=1, N=2999,
                                            integration_domain=[[1e-10, 10]])  # shape : (n,)
                    log_a = torch.log(intg_a)

                    return (log_a).reshape(-1,1)
                    # log0 =-torch.log(2*gamma_func(1+1/alpha))
                    # return (-torch.abs(x)**alpha+log0)
            elif noise_mode == 'independent':
                if dimension==1:
                    return torch.log(levy.pdf(x, alpha))
                else:
                    return torch.log(levy.pdf(x, alpha)).sum(dim=1, keepdim=True)

        def ode_solver_func(t, x):
            sample = torch.from_numpy(x[:-shape[0]].reshape((shape[0], shape[1]))).to(device).float()
            t_array = (torch.ones((shape[0], 1)).to(device) * t).requires_grad_(False)
            drift = drift_fn(score_model, sample, t_array).detach()
            logp_grad = div_fn(score_model, sample,t_array, epsilon)
            return torch.cat([drift.reshape((-1,)), logp_grad.reshape((-1,))], axis=0).cpu().numpy()

        init = torch.cat([data.reshape((-1,)), torch.zeros((shape[0],))], axis=0).reshape((-1,)).cpu().numpy()

        solution = integrate.solve_ivp(ode_solver_func, (t_0, T), init, rtol=rtol, atol=atol, method=method)
        nfe = solution.nfev
        t = solution.t
        zp = solution.y[:, -1]
        z =torch.tensor( zp[:-shape[0]].reshape(shape)).to('cuda').float()
        prior = prior_log(z).cpu().numpy()

        logpx = zp[-shape[0]:].reshape((shape[0], 1))+prior


        return logpx

    return likelihood_fn