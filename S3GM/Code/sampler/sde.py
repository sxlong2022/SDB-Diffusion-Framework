import abc
import torch
import numpy as np


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, net_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

    Args:
      net_fn: a z-dependent PFGM that takes x and z and returns the normalized Poisson field.
        Or a time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""

                drift, diffusion = sde_fn(x, t)
                score = net_fn(x.float(), t.float())
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = torch.zeros_like(diffusion) if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * net_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VESDE(SDE):
    def __init__(self, config):
        """Construct a Variance Exploding SDE.

        Args:
            config: Configuration object, must contain the following parameters:
                - sigma_min: smallest sigma
                - sigma_max: largest sigma
                - N: number of discretization steps
        """
        # Get parameters from config
        sigma_min = config.get('sigma_min', 0.01)
        sigma_max = config.get('sigma_max', 2.0)
        N = config.get('N', 1000)
        
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N
        self.config = config

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (
                    2 * self.sigma_max ** 2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G


class VPSDE(SDE):
    def __init__(self, config):
        """Construct VPSDE

        Args:
            config: Configuration object containing the following parameters:
                - beta_min: value of beta(0), default 0.1
                - beta_max: value of beta(1), default 20.0
                - num_scales: discretization steps (corresponds to SDE's N)
        """
        # --- Read parameters from passed config object ---
        beta_min = config.beta_min # Direct attribute access
        beta_max = config.beta_max # Direct attribute access
        N = config.num_scales      # Use num_scales as N

        super().__init__(N)

        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.config = config # Store complete config object

        # Recalculate discretization sequences
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Numerical stability parameters
        self.stability_eps = 1e-8

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        """Compute SDE drift and diffusion coefficients
        
        Args:
            x: Input tensor
            t: Timestep, range [0,1]
        """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """Compute marginal probability distribution parameters
        
        Args:
            x: Input tensor
            t: Timestep
        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        
        # Adjust coefficient shape based on input dimensions
        if len(x.shape) == 4:  # [B,C,H,W]
            mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        elif len(x.shape) == 5:  # [B,T,C,H,W]
            mean = torch.exp(log_mean_coeff[:, None, None, None, None]) * x
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        """Sample from prior distribution
        
        Args:
            shape: Sampling shape
        """
        return torch.randn(*shape)

    def prior_logp(self, z):
        """Compute log probability density of prior distribution"""
        shape = z.shape
        N = np.prod(shape[1:])
        logp = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logp

    def discretize(self, x, t, channel_modal=None):
        """DDPM discretization."""
        if isinstance(t, float):
            t = torch.full((x.shape[0],), t, device=x.device)
        timestep = (t * (self.N - 1)).long()
        beta = self.discrete_betas.to(t.device)[timestep]
        alpha = self.alphas.to(t.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        std = self.sqrt_1m_alphas_cumprod.to(t.device)[timestep] + 1e-8
        
        G = sqrt_beta

        f = torch.sqrt(beta)[:, None, None, None, None] * x
        
        return f, G
