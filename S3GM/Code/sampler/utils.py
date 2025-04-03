import torch
import numpy as np
import abc
import functools
from tqdm import tqdm
from scipy import integrate
from sampler.sde import VESDE, VPSDE
from trainer.loss import predict_fn, voriticity_residual, sample_noise, kse_residual
from einops import rearrange
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)

class ODE_Solver(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, net_fn, eps=None):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        if sde.config.sde != 'poisson':
            self.rsde = sde.reverse(net_fn, probability_flow=True)
        self.net_fn = net_fn
        self.eps = eps

    @abc.abstractmethod
    def update_fn(self, x, t, t_list=None, idx=None):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, net_fn, probability_flow=False, channel_modal=None, eps=None):
        super().__init__()
        self.sde = sde
        self.channel_modal = channel_modal
        # Compute the reverse SDE/ODE
        if sde.config.sde != 'poisson':
            self.rsde = sde.reverse(net_fn, probability_flow)
        self.net_fn = net_fn
        self.eps = eps

    @abc.abstractmethod
    def update_fn(self, x, t, t_list=None, idx=None):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, net_fn, snr, n_steps, channel_modal=None):
        super().__init__()
        self.sde = sde
        self.net_fn = net_fn
        self.snr = snr
        self.n_steps = n_steps
        self.channel_modal = channel_modal

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, net_fn, probability_flow=False, eps=None):
        super().__init__(sde, net_fn, probability_flow, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        z = torch.randn_like(x)
        if self.sde.config.sde == 'poisson':
            if t_list is None:
                dt = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
            else:
                # integration over z
                dt = - (1 - torch.exp(t_list[idx + 1] - t_list[idx]))
                dt = float(dt.cpu().numpy())
            drift = self.sde.ode(self.net_fn, x, t)
            diffusion = torch.zeros((len(x)), device=x.device)
        else:
            if t_list is None:
                dt = -1. / self.sde.N
            drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


class ForwardEulerPredictor(ODE_Solver):
    def __init__(self, sde, net_fn, eps=None):
        super().__init__(sde, net_fn, eps)

    def update_fn(self, x, t, t_list=None, idx=None):

        if self.sde.config.sde == 'poisson':
            # dt = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
            drift = self.sde.ode(self.net_fn, x, t)
            if t_list is None:
                dt = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
            else:
                # integration over z
                dt = - (1 - torch.exp(t_list[idx + 1] - t_list[idx]))
                dt = float(dt.cpu().numpy())
        else:
            dt = -1. / self.sde.N
            drift, _ = self.rsde.sde(x, t)
        x = x + drift * dt
        return x


class ImprovedEulerPredictor(ODE_Solver):
    def __init__(self, sde, net_fn, eps=None):
        super().__init__(sde, net_fn, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        if self.sde.config.sde == 'poisson':
            if t_list is None:
                dt = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
            else:
                # integration over z
                dt = (torch.exp(t_list[idx + 1] - t_list[idx]) - 1)
                dt = float(dt.cpu().numpy())
            drift = self.sde.ode(self.net_fn, x, t)
        else:
            dt = -1. / self.sde.N
            drift, _ = self.rsde.sde(x, t)
        x_new = x + drift * dt

        if idx == self.sde.N - 1:
            return x_new
        else:
            idx_new = idx + 1
            t_new = t_list[idx_new]
            t_new = torch.ones(len(t), device=t.device) * t_new

            if self.sde.config.sde == 'poisson':
                if t_list is None:
                    dt_new = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
                else:
                    # integration over z
                    dt_new = (1 - torch.exp(t_list[idx] - t_list[idx + 1]))
                    dt_new = float(dt_new.cpu().numpy())
                drift_new = self.sde.ode(self.net_fn, x_new, t_new)
            else:
                drift_new, diffusion = self.rsde.sde(x_new, t_new)
                dt_new = -1. / self.sde.N

            x = x + (0.5 * drift * dt + 0.5 * drift_new * dt_new)
            return x


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, net_fn, probability_flow=False, channel_modal=None, eps=None):
        super().__init__(sde, net_fn, probability_flow, channel_modal, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        t_shape = t.shape
        f, G = self.rsde.discretize(x, t, self.channel_modal)
        z = torch.randn_like(x)
        x_mean = x - f
        G = G.view(*t_shape)
        if self.channel_modal is None:
            x = x_mean + G[:, None, None, None] * z
        else:
            G = G.repeat_interleave(torch.tensor(self.channel_modal).to(G.device), dim=1)
            x = x_mean + G[:, :, None, None] * z
        return x, x_mean


class ReverseDiffusionPredictorMM(Predictor):
    def __init__(self, sde, net_fn, probability_flow=False, channel_modal=None, eps=None):
        super().__init__(sde, net_fn, probability_flow, channel_modal, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        t_shape = t.shape
        f, G = self.rsde.discretize(x, t, self.channel_modal)
        z = sample_noise(x.shape, channel_modal=self.channel_modal, device=x.device, dtype=x.dtype)
        x_mean = x - f
        G = G.view(*t_shape)
        if self.channel_modal is None:
            x = x_mean + G[:, None, None, None] * z
        else:
            G = G.repeat_interleave(torch.tensor(self.channel_modal).to(G.device), dim=1)
            x = x_mean + G[:, :, None, None] * z
        return x, x_mean


class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, net_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, t_list=None, idx=None):
        return x, x



class LangevinCorrector(Corrector):
  def __init__(self, sde, net_fn, snr, n_steps, channel_modal=None):
    super().__init__(sde, net_fn, snr, n_steps, channel_modal)
    if not isinstance(sde, VPSDE) \
        and not isinstance(sde, VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    net_fn = self.net_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, VPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      # Clamp timestep to avoid index out of bounds
      timestep = torch.clamp(timestep, 0, sde.N - 1)
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      # For VESDE, alpha is effectively 1 in this context
      alpha = torch.ones_like(t).to(t.device)

    # Store original x for comparison if needed
    x_orig = x.clone()

    # --- Determine if we should log based on the main loop step (approximated by t) ---
    # Calculate approximate main loop step index 'main_i'
    current_t = t.mean().item()
    # Avoid division by zero if sde.T is zero or t equals sde.T exactly in the first step
    if sde.T > 0 and current_t < sde.T: 
        main_i_approx = round(sde.N * (1 - current_t / sde.T))
    else:
        main_i_approx = 0 # Assume step 0 if t is sde.T or sde.T is 0
        
    # Log only if main_i is approximately a multiple of 100 or the last step
    should_log_corrector_details = (main_i_approx % 100 == 0) or (main_i_approx >= sde.N - 1)
    # --- End logging frequency check ---

    for i in range(n_steps):
      grad = net_fn(x, t)
      noise = torch.randn_like(x)

      grad_flat = grad.reshape(grad.shape[0], -1)
      noise_flat = noise.reshape(noise.shape[0], -1)

      epsilon = 1e-6
      grad_norm = torch.norm(grad_flat, dim=-1) + epsilon
      noise_norm = torch.norm(noise_flat, dim=-1) + epsilon

      snr_ratio_sq = (target_snr * noise_norm / grad_norm) ** 2
      step_size = snr_ratio_sq * 2 * alpha

      step_size_val = step_size.mean().item()
      grad_norm_val = grad_norm.mean().item()
      noise_norm_val = noise_norm.mean().item()
      snr_ratio_sq_val = snr_ratio_sq.mean().item()

      step_size_clamp_max = 0.1 # Tunable parameter
      step_size = torch.clamp(step_size, min=1e-8, max=step_size_clamp_max)
      step_size_clamped_val = step_size.mean().item()

      # --- Conditional Logging ---
      if should_log_corrector_details:
          log_level = logging.INFO # Or logging.DEBUG for more detail
          logger.log(log_level, f"  Corrector Step {i} (t={current_t:.4f}, approx_main_step={main_i_approx}):")
          logger.log(log_level, f"    grad_norm={grad_norm_val:.4e}, noise_norm={noise_norm_val:.4e}")
          logger.log(log_level, f"    snr_ratio_sq={snr_ratio_sq_val:.4e}, alpha={alpha.mean().item():.4e}")
          logger.log(log_level, f"    step_size_unclamped={step_size_val:.4e}")
          logger.log(log_level, f"    step_size_clamped={step_size_clamped_val:.4e} (max={step_size_clamp_max})")
      # ---------------------------

      step_size_sqrt_term = torch.sqrt(step_size * 2)

      if len(x.shape) > 4: # Handle 5D tensor (video)
        step_size_exp = step_size[:, None, None, None, None]
        step_size_sqrt_exp = step_size_sqrt_term[:, None, None, None, None]
      else: # Handle 4D tensor
        step_size_exp = step_size[:, None, None, None]
        step_size_sqrt_exp = step_size_sqrt_term[:, None, None, None]

      if torch.isnan(grad).any() or torch.isnan(step_size_exp).any() or torch.isnan(step_size_sqrt_exp).any():
           logger.error(f"NaN detected in Corrector update inputs at step {i} (t={current_t:.4f}). Skipping update.")
           continue

      x_mean = x + step_size_exp * grad
      x = x_mean + step_size_sqrt_exp * noise

      if torch.isnan(x).any():
          logger.error(f"NaN detected after Corrector update step {i} (t={current_t:.4f}). Reverting to previous state.")
          x = x_orig
          break

    return x, x_mean


class LangevinCorrectorMM(Corrector):
    def __init__(self, sde, net_fn, snr, n_steps, channel_modal=None):
        super().__init__(sde, net_fn, snr, n_steps, channel_modal)
        if not isinstance(sde, VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        net_fn = self.net_fn
        n_steps = self.n_steps
        target_snr = self.snr

        if isinstance(sde, VESDE):
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = net_fn(x, t)
            noise = sample_noise(x.shape, channel_modal=self.channel_modal, device=x.device, dtype=x.dtype)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            if self.channel_modal is None:
                x_mean = x + step_size[:, None, None, None] * grad
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise
            else:
                step_size = step_size.repeat_interleave(torch.tensor(self.channel_modal).to(step_size.device), dim=1)
                x_mean = x + step_size[:, :, None, None] * grad
                x = x_mean + torch.sqrt(step_size * 2)[:, :, None, None] * noise

        return x, x_mean


class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

    def __init__(self, sde, net_fn, snr, n_steps):
        super().__init__(sde, net_fn, snr, n_steps)
        if not isinstance(sde, VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        net_fn = self.net_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VESDE):
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = net_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, net_fn, snr, n_steps, channel_modal=None):
        pass

    def update_fn(self, x, t):
        return x, x


def shared_ode_solver_update_fn(x, t, sde, net, ode_solver, eps, t_list=None, idx=None):
    """A wrapper that configures and returns the update function of ODE solvers."""
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    ode_solver_obj = ode_solver(sde, net_fn, eps)
    return ode_solver_obj.update_fn(x, t, t_list=t_list, idx=idx)


def shared_predictor_update_fn(x, t, sde, net, predictor, probability_flow, continuous, eps,
                               channel_modal=None, t_list=None, idx=None):
    """A wrapper that configures and returns the update function of predictors."""
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, net_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, net_fn, probability_flow, channel_modal, eps)
    return predictor_obj.update_fn(x, t, t_list=t_list, idx=idx)


def shared_corrector_update_fn(x, t, sde, net, corrector, continuous, snr, n_steps, channel_modal=None):
    """A wrapper tha configures and returns the update function of correctors."""
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, net_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, net_fn, snr, n_steps, channel_modal=channel_modal)
    return corrector_obj.update_fn(x, t)


def ode_sampler(net, sde, ode_solver, shape, device='cpu', dtype='float32', eps=1e-3):
    ode_update_fn = functools.partial(shared_ode_solver_update_fn,
                                      sde=sde,
                                      ode_solver=ode_solver,
                                      eps=eps)
    x = sde.prior_sampling(shape).to(device).float()
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()

    xs = []
    for i in tqdm(range(sde.N), desc='generating...', total=sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device).float() * t
        x = ode_update_fn(x, vec_t, net=net, t_list=timesteps, idx=i)
        xs.append(x)
    return x, sde.N


def pc_sampler(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3):
    dtype_torch = getattr(torch, dtype)
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous,
                                            eps=eps,
                                            channel_modal=config.channel_modal)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            channel_modal=config.channel_modal)
    # if is_mm:
    #     x = sample_noise(shape, channel_modal=config.channel_modal, device=device, dtype=dtype_torch)*sde.sigma_max
    # else:
    x = sde.prior_sampling(shape).to(device).float()
    x0 = torch.tensor(x0, device=device).float()
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()
    # if 'mm' in config.version:
    #     mode = np.array(config.mm_mode)
    #     mode_r = torch.tensor(mode.repeat(config.channel_modal, 0), device=device).bool()

    x_generated = [x.detach().cpu().numpy()]
    for i in tqdm(range(sde.N)):
        t = timesteps[i]
        # if 'mm' in config.version:
        #     t = torch.tensor(util.mode_to_ts(mode, pos=eps_t, neg=t), device=device).float()
        #     vec_t = torch.ones([shape[0], config.num_modals], device=t.device).float() * t[None, :]
        #     x = x * (~mode_r)[None, :, None, None] + x0 * mode_r[None, :, None, None]
        #     if 'cond' in config.version:
        #         vec_t = torch.cat([vec_t, torch.ones_like(vec_t[:, :1])*pattern], dim=1)
        # else:
        vec_t = torch.ones(shape[0], device=t.device).float() * t
        x, x_mean = corrector_update_fn(x, vec_t, net=net)
        # if 'mm' in config.version:
        #     x = x * (~mode_r)[None, :, None, None] + x0 * mode_r[None, :, None, None]
        x, x_mean = predictor_update_fn(x, vec_t, net=net)
        x_generated.append(x_mean.detach().cpu().numpy())

    return x_mean if denoise else x, x_generated


def pc_sampler_video_ar(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False, alpha=1.,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            channel_modal=config.channel_modal)
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    x0 = torch.tensor(x0, device=device).float()            # batch*ol*(c+npara)*h*w
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()

    nf = config.num_frames
    ns = config.num_steps
    ncomp = config.num_components
    ol = config.overlap
    b = int(ns//(nf-ol)+1)      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [config.num_samples, nf, ncomp+config.num_modals-1, config.image_size, config.image_size]       # batch*nf*(c+npara)*h*w

    transform = lambda x: x[:, :ol]

    x_generated = []
    pred = []
    for i_b in range(b):
        if i_b == 0:
            y = x0
        else:
            y = x_mean[:, -ol:].detach()
        x = sde.prior_sampling(shape).to(device).float()
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device).float() * t
            '''corrector'''
            x, x_mean = corrector_update_fn(x, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            with torch.enable_grad():
                inp = x.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)
                with torch.no_grad():
                    f, G = sde.discretize(x, vec_t)
                    rev_f = f - G[:, None, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
                    rev_G = torch.zeros_like(G) if probability_flow else G
                    x_mean = x - rev_f
                    x_u = x_mean + rev_G[:, None, None, None, None] * z
                    # x_u = x
                # dps loss
                _, std = sde.marginal_prob(x, t)
                x0_hat = std**2*score + inp
                loss_dps = ((y-transform(x0_hat))**2).sum()
                loss = alpha * loss_dps
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                x = x_u - dx       # /torch.sqrt(scalar1.mean())
                # x = x_u
            x = x.detach()
        pred.append(x_mean if i_b==0 else x_mean[:, ol:])
    return torch.cat(pred, dim=1).detach().cpu().numpy() if denoise else x, x_generated


def pc_sampler_video1d_ar(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False, alpha=1.,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            channel_modal=config.channel_modal)
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    x0 = torch.tensor(x0, device=device).float()            # batch*ol*(c+npara)*h*w
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()

    nf = config.num_frames
    ns = config.num_steps
    ncomp = config.num_components
    ol = config.overlap
    b = int(ns//(nf-ol)+1)      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [config.num_samples, nf, ncomp+config.num_modals-1, config.image_size]       # batch*nf*(c+npara)*h

    transform = lambda x: x[:, :ol]

    x_generated = []
    pred = []
    for i_b in range(b):
        if i_b == 0:
            y = x0
        else:
            y = x_mean[:, -ol:].detach()
        x = sde.prior_sampling(shape).to(device).float()
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device).float() * t
            '''corrector'''
            x, x_mean = corrector_update_fn(x, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            with torch.enable_grad():
                inp = x.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)
                with torch.no_grad():
                    f, G = sde.discretize(x, vec_t)
                    rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
                    rev_G = torch.zeros_like(G) if probability_flow else G
                    x_mean = x - rev_f
                    x_u = x_mean + rev_G[:, None, None, None] * z
                    # x_u = x
                # dps loss
                _, std = sde.marginal_prob(x, t)
                x0_hat = std**2*score + inp
                loss_dps = ((y-transform(x0_hat))**2).sum()
                loss = alpha * loss_dps
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                x = x_u - dx       # /torch.sqrt(scalar1.mean())
                # x = x_u
            x = x.detach()
        pred.append(x_mean if i_b==0 else x_mean[:, ol:])
    return torch.cat(pred, dim=1).detach().cpu().numpy() if denoise else x, x_generated


def complete_video_pc_dps(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=1., gamma1=100., gamma2=100, snr=0.128, std_y=None, gamma=1.e-2,
                         device='cpu', dtype='float32', eps=1e-3, save_sample_path=False,
                         probability_flow=False, continuous=True, data_scalar=None):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            channel_modal=config.channel_modal)
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    # x_known = torch.from_numpy(x0).to(device).type(dtype_torch)
    y = torch.from_numpy(y).to(device).type(dtype_torch)
    # shape_sample = [len(y), config.num_channels, config.image_size, config.image_size]

    nf = config.num_frames
    ns = config.num_steps
    ncomp = config.num_components
    ol = config.overlap
    b = max(1, int(ns // max(1, (nf - ol))) + 1)  # 防止除零
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    # 注意: 这里shape的第二维应该是b，而不是ns_real
    shape = [config.num_samples, b, nf, ncomp+config.num_modals-1, config.image_size, config.image_size]       # batch*b*nf*(c+npara)*h*w
    shape_sample = [config.num_samples, ns_real, ncomp+config.num_modals-1, config.image_size, config.image_size]     # batch*ns_real*(c+npara)*h*w

    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([config.num_samples, ns_real, ncomp+config.num_modals-1, config.image_size, config.image_size], dtype=dtype_torch, device=device)   # batch*ns_real*(c+npara)*h*w
        for i in range(b):
            i_inv = b - i - 1
            # 确保索引不越界
            start_dest = i_inv * (nf - ol)
            end_dest = start_dest + nf
            if start_dest < ns_real: # 检查起始索引
                # 计算实际要复制的帧数
                num_frames_to_copy = min(nf, ns_real - start_dest)
                # 检查源索引
                if i_inv < xx.shape[1]:
                    # 复制数据通道
                    sample[:, start_dest : start_dest + num_frames_to_copy, :ncomp] = xx[:, i_inv, :num_frames_to_copy, :ncomp]
                else:
                    logger.warning(f"x_to_sample: 索引 {i_inv} 超出 xx 的第二维范围 {xx.shape[1]}")
            else:
                 logger.warning(f"x_to_sample: 起始索引 {start_dest} 超出 sample 的范围 {ns_real}")

        # 检查参数通道的源索引
        if 0 < xx.shape[1]: # 确保第一批次存在
             # --- 修改 expand 的 size 参数 ---
             # 源张量 xx[:, 0:1, 0:1, ncomp:] 的形状是 [B, 1, 1, C_param, H, W] (6维)
             # 目标形状是 [B, ns_real, C_param, H, W]
             # expand 需要 6 个维度参数
             # 第0维 (Batch): -1 (保持不变)
             # 第1维 (b dim): N/A -> 扩展到 ns_real (但源是1)
             # 第2维 (Time): 1 -> 扩展到 ns_real
             # 第3维 (Channel): C_param -> -1 (保持不变)
             # 第4维 (Height): H -> -1 (保持不变)
             # 第5维 (Width): W -> -1 (保持不变)
             # 注意：expand 不能改变元素数量，它通过重复现有维度来扩展。
             # 我们需要从 xx 的第一个 batch (b=0) 和第一个时间步 (t=0) 获取参数通道
             param_source = xx[:, 0:1, 0:1, ncomp:] # Shape: [B, 1, 1, C_param, H, W]
             # 扩展时间维度和 batch 维度 (如果 B > 1，虽然这里 B=1)
             # 目标 sample[:, :, ncomp:] 形状是 [B, ns_real, C_param, H, W]
             # expand 需要 6 个维度来匹配 param_source
             # [B, ns_real, 1, C_param, H, W] ? 不对，expand 不能插入维度
             # 应该先 squeeze 再 expand?
             # param_source.squeeze(1).squeeze(1) -> [B, C_param, H, W] (4维)
             # sample[:, :, ncomp:] 是 5 维

             # --- 尝试直接赋值和广播 ---
             # 获取源参数通道 [B, 1, C_param, H, W]
             param_source_frame0 = xx[:, 0, 0:1, ncomp:]
             # param_source_frame0 的形状应该是 [B, 1, C_param, H, W]
             # sample[:, :, ncomp:] 的形状是 [B, ns_real, C_param, H, W]
             # PyTorch 的广播应该能处理这个问题，只要维度兼容
             sample[:, :, ncomp:] = param_source_frame0 # 自动广播时间维度

             # --- 原来的 expand 代码 (注释掉) ---
             # sample[:, :, ncomp:] = xx[:, 0:1, 0:1, ncomp:].expand(-1, ns_real, -1, -1, -1) # 扩展参数通道 (错误)
             # ------------------------------------
        else:
            logger.warning(f"x_to_sample: xx 的第二维为空，无法复制参数通道")

        return sample

    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()
    x_unknown = sde.prior_sampling(shape).to(device).float()    # batch*b*nf*(c+npara)*h*w
    
    x = x_unknown           # batch*b*(nf*c+npara)*h*w
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []

    # --- 添加辅助函数用于打印范围和标准差 ---
    def print_stats(tensor, name):
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            print(f"  {name}: N/A")
            return
        try:
            tensor_finite = tensor[torch.isfinite(tensor)]
            if tensor_finite.numel() > 0:
                 print(f"  {name}: "
                       f"Range=[{tensor_finite.min().item():.4e}, {tensor_finite.max().item():.4e}], "
                       f"Std={tensor_finite.std().item():.4e}, "
                       f"NaNs={torch.isnan(tensor).sum().item()}, Infs={torch.isinf(tensor).sum().item()}")
            else:
                 print(f"  {name}: All values are NaN/Inf. NaNs={torch.isnan(tensor).sum().item()}, Infs={torch.isinf(tensor).sum().item()}")
        except Exception as e:
            print(f"  Error printing stats for {name}: {e}")
    # --------------------------------------

    # 添加数据范围监控函数
    def monitor_data_range(tensor, name, step=None):
        # ... (保持原样) ...
        pass # 暂时禁用，使用 print_stats

    monitor_data_range(x, "初始输入")

    with torch.enable_grad():
        pbar = tqdm(range(sde.N), desc="采样进度")
        for i in pbar:
            t = timesteps[i]
            vec_t = torch.ones(shape[0]*b, device=t.device).float() * t

            # --- 调试日志: 循环开始 ---
            if i == 0 or i == 25 or i == sde.N -1 : # 只在特定步骤打印详细信息
                print(f"\n--- Step {i}, t={t.item():.4f} ---")
                print_stats(x, "x (循环开始)")

            '''method 1 (batched)'''
            xb = rearrange(x, 'b n t c h w -> (b n) t c h w')       # (batch*b)*nf*(c+npara)*h*w

            '''corrector'''
            # --- 调试日志: Corrector 之前 ---
            if i == 0 or i == 25 or i == sde.N -1:
                print_stats(xb, "xb (Corrector 输入)")
            temp, temp_mean_corrector = corrector_update_fn(xb, vec_t, net=net)     # (batch*b)*nf*(c+npara)*h*w
            # --- 调试日志: Corrector 之后 ---
            if i == 0 or i == 25 or i == sde.N -1:
                print_stats(temp, "temp (Corrector 输出)")
                print_stats(temp_mean_corrector, "temp_mean (Corrector 输出)")

            '''predictor'''
            z = torch.randn_like(x)
            zb = rearrange(z, 'b n t c h w -> (b n) t c h w')

            inp = temp.clone()                  # (batch*b)*nf*(c+npara)*h*w
            inp.requires_grad_(True)

            # --- 调试日志: Predictor/DPS 之前 ---
            if i == 0 or i == 25 or i == sde.N -1:
                print_stats(inp, "inp (Predictor 输入)")

            score = net_fn(inp, vec_t)          # (batch*b)*nf*(c+npara)*h*w
            # --- 调试日志: Score ---
            if i == 0 or i == 25 or i == sde.N -1:
                print_stats(score, "score (模型输出)")

            with torch.no_grad():
                # --- 调试日志: SDE 系数计算之前 ---
                if i == 0 or i == 25 or i == sde.N -1:
                    print_stats(temp, "temp (SDE 系数输入)")
                f, G = sde.discretize(temp, vec_t)
                # --- 调试日志: SDE 系数 ---
                if i == 0 or i == 25 or i == sde.N -1:
                    print_stats(f, "f (SDE drift)")
                    print_stats(G, "G (SDE diffusion)")
                score_detached = score.detach() # 使用 detach 后的 score 计算
                rev_f = f - G[:, None, None, None, None] ** 2 * score_detached * (0.5 if probability_flow else 1.)
                rev_G = torch.zeros_like(G) if probability_flow else G
                temp_mean_predictor = temp - rev_f # Predictor 的 x_mean
                temp_u = temp_mean_predictor + rev_G[:, None, None, None, None] * zb # Predictor 的 x
                # --- 调试日志: Predictor 计算后 ---
                if i == 0 or i == 25 or i == sde.N -1:
                    print_stats(rev_f, "rev_f")
                    print_stats(temp_mean_predictor, "temp_mean (Predictor 输出)")
                    print_stats(temp_u, "temp_u (Predictor+Noise 输出)")

            # dps loss
            _, std = sde.marginal_prob(xb, vec_t)
            # --- 调试日志: SDE std ---
            if i == 0 or i == 25 or i == sde.N -1:
                 print_stats(std, "std (SDE marginal)")
            # -------------------------

            stability_eps = 1e-8

            if isinstance(sde, VPSDE):
                # --- 修正 VPSDE x0_hat 计算 ---
                # 使用 sde.alphas_cumprod 计算 sqrt_alpha_t
                # 获取对应时间步的 alphas_cumprod
                discrete_t_indices = (vec_t * (sde.N - 1)).long().clamp(0, sde.N - 1) # 将连续时间映射到离散索引
                alphas_cumprod_t = sde.alphas_cumprod.to(vec_t.device)[discrete_t_indices]
                sqrt_alpha_t = torch.sqrt(alphas_cumprod_t) + stability_eps
                sqrt_1m_alpha_t = torch.sqrt(1.0 - alphas_cumprod_t) + stability_eps # 这是噪声的标准差 std

                # 扩展维度
                sqrt_alpha_t_exp = sqrt_alpha_t[:, None, None, None, None]
                sqrt_1m_alpha_t_exp = sqrt_1m_alpha_t[:, None, None, None, None]

                # --- 调试日志: VPSDE 系数 ---
                if i == 0 or i == 25 or i == sde.N -1:
                    print_stats(sqrt_alpha_t_exp, "sqrt_alpha_t")
                    print_stats(sqrt_1m_alpha_t_exp, "sqrt_1m_alpha_t (std)")
                # --------------------------

                # 使用 clamp 限制 score 范围
                score_clamp = torch.clamp(score, min=-config.stability['score_clamp_range'], max=config.stability['score_clamp_range'])
                # 计算 x0_hat
                x0_hat_calc = (inp - sqrt_1m_alpha_t_exp * score_clamp) / sqrt_alpha_t_exp

                # --- Adjustment: Widen x0_hat clamp range ---
                # 如果启用了 x0_hat clamp (现在放宽范围)
                if config.stability.get('x0_hat_clamp', True):
                     # 将范围从 [-10, 10] 调整为 [-20, 20]
                     x0_hat_calc = torch.clamp(x0_hat_calc, -20.0, 20.0) 
                # --- End Adjustment ---

                x0_hat = rearrange(x0_hat_calc, '(b n) t c h w -> b n t c h w', n=b)
            else: # VESDE 或其他
                std_exp = std[:, None, None, None, None] if len(inp.shape) == 5 else std[:, None, None, None]
                score_clamp = torch.clamp(score, min=-config.stability['score_clamp_range'], max=config.stability['score_clamp_range'])
                x0_hat_calc = std_exp ** 2 * score_clamp + inp
                
                # --- Adjustment: Widen x0_hat clamp range ---
                # 如果启用了 x0_hat clamp (现在放宽范围)
                if config.stability.get('x0_hat_clamp', True):
                     # 将范围从 [-10, 10] 调整为 [-20, 20]
                     x0_hat_calc = torch.clamp(x0_hat_calc, -20.0, 20.0) 
                # --- End Adjustment ---
                     
                x0_hat = rearrange(x0_hat_calc, '(b n) t c h w -> b n t c h w', n=b)

            # --- 调试日志: x0_hat ---
            if i == 0 or i == 25 or i == sde.N -1:
                print_stats(x0_hat, "x0_hat (估算)")
            # ------------------------

            if torch.isnan(x0_hat).any() or torch.isinf(x0_hat).any():
                print(f"警告: 步骤 {i}, x0_hat 计算产生了 NaN 或 Inf 值。替换为0。")
                x0_hat = torch.nan_to_num(x0_hat, nan=0.0, posinf=5.0, neginf=-5.0)

            x0_hat_temp = x_to_sample(x0_hat)
            if save_sample_path:
                x0_hats.append(x0_hat.detach().cpu().numpy())

            # --- 计算 DPS 损失 ---
            var = std_y**2 + gamma * std**2 if std_y is not None else 1.
            # 扩展 var 的维度以匹配
            if isinstance(var, float) or (isinstance(var, torch.Tensor) and var.dim() == 0):
                # var 是标量（float 或 0维张量）
                var_exp = torch.tensor(var, device=device) # 转换为张量
            else:
                # var 是多维张量
                var_exp = var[:, None, None, None, None] if len(y.shape) == 5 else var[:, None, None, None]
            # 添加 epsilon 防止除零
            var_safe = var_exp + stability_eps
            # 计算损失项
            loss_dps_term = (y - transform(x0_hat_temp)) ** 2 / var_safe
            # 检查损失项中的 NaN/Inf
            if torch.isnan(loss_dps_term).any() or torch.isinf(loss_dps_term).any():
                 print(f"警告: 步骤 {i}, loss_dps_term 包含 NaN/Inf。替换为0。")
                 loss_dps_term = torch.nan_to_num(loss_dps_term, nan=0.0) # Inf 也会被处理

            loss_dps = torch.sum(loss_dps_term.reshape(x0_hat.shape[0], -1), dim=-1)
            loss_dps = torch.sum(loss_dps, dim=0)
            if std_y is not None:
                loss_dps = loss_dps / 2.
            # --- 调试日志: loss_dps ---
            if i == 0 or i == 25 or i == sde.N -1:
                 print(f"  loss_dps: {loss_dps.item():.4e}")
            # --------------------------

            # --- 计算一致性损失 ---
            if b == 1:
                x0_curr = x0_hat[:, 0, :(nf-1), :ncomp].detach()
                x0_next = x0_hat[:, 0, 1:nf, :ncomp]
                loss_consis_term = (x0_curr - x0_next)**2
                loss_consis_para_term = (x0_hat[:, 0, 1:, ncomp:] - x0_hat[:, 0, :1, ncomp:].detach())**2
            else:
                x0_curr = x0_hat[:, :-1, (nf-ol):nf, :ncomp].detach()
                x0_next = x0_hat[:, 1:, :ol, :ncomp]
                loss_consis_term = (x0_curr - x0_next)**2
                loss_consis_para_term = (x0_hat[:, 1:, :, ncomp:] - x0_hat[:, 0:1, :, ncomp:].detach())**2

            loss_consis = torch.sum(loss_consis_term)
            loss_consis_para = torch.sum(loss_consis_para_term)

            # --- 调试日志: loss_consis & loss_consis_para ---
            if i == 0 or i == 25 or i == sde.N -1:
                 print(f"  loss_consis: {loss_consis.item():.4e}")
                 print(f"  loss_consis_para: {loss_consis_para.item():.4e}")
            # -----------------------------------------------

            loss_eq = torch.tensor(0.0, device=device) # 默认值
            if config.physics_guide:
                # loss_eq, scalar2 = voriticity_residual(x0_hat, ns_real, 1., data_scalar) # 这部分可能需要调整
                # scalar2 = scalar2.detach()
                # loss = alpha * loss_dps + beta * loss_eq + gamma1 * loss_consis + gamma2 * loss_consis_para
                # assert (not torch.isnan(loss_eq))
                pass # 暂时跳过物理引导损失

            # 归一化损失值，防止过大
            loss_dps_norm = loss_dps / (loss_dps.detach().abs().mean() + 1.0)
            loss_consis_norm = loss_consis / (loss_consis.detach().abs().mean() + 1.0)
            loss_consis_para_norm = loss_consis_para / (loss_consis_para.detach().abs().mean() + 1.0)

            # 使用归一化后的损失
            loss = alpha * loss_dps_norm + gamma1 * loss_consis_norm + gamma2 * loss_consis_para_norm

            # --- 调试日志: total loss ---
            if i == 0 or i == 25 or i == sde.N -1:
                 print(f"  loss_total: {loss.item():.4e}")
            # ----------------------------

            # --- 梯度计算与应用 ---
            dx = torch.autograd.grad(loss, inp, allow_unused=True)[0] # 允许未使用的梯度

            if dx is None:
                print(f"警告: 步骤 {i}, dx 为 None，跳过梯度更新。")
                temp = temp_u # 如果梯度为None，则不应用梯度更新
            else:
                # 检查 dx 是否包含 NaN/Inf
                if torch.isnan(dx).any() or torch.isinf(dx).any():
                    print(f"警告: 步骤 {i}, dx 包含 NaN/Inf。替换为0。")
                    dx = torch.nan_to_num(dx, nan=0.0) # Inf 也会被处理

                # --- 调试日志: dx ---
                if i == 0 or i == 25 or i == sde.N -1:
                    print_stats(dx, "dx (梯度)")

                # 限制梯度大小
                dx = torch.clamp(dx, min=-1e5, max=1e5) # 进一步限制梯度范围

                temp = temp_u - dx     # (batch*b)*(nf*c+npara)*h*w

            # --- 调试日志: temp (最终) ---
            if i == 0 or i == 25 or i == sde.N -1:
                print_stats(temp, "temp (最终更新)")

            temp = temp.detach() # detach 在这里
            x = rearrange(temp, '(b n) t c h w -> b n t c h w', n=b)
            x_mean = rearrange(temp_mean_corrector, '(b n) t c h w -> b n t c h w', n=b)

            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())

            # 更新 tqdm 描述
            pbar.set_description(f"采样进度 (Loss: {loss.item():.2e})")

    # --- 在函数末尾添加最终结果的监控 ---
    final_result = x_to_sample(x_mean).detach().cpu().numpy()
    print("\n--- 采样完成 ---")
    print_stats(torch.from_numpy(final_result), "最终结果 x_mean (转换后)")

    return final_result, x_generated if save_sample_path else None


def complete_video1d_pc_dps(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=None, gamma1=100., gamma2=100, snr=0.128, std_y=None, gamma=1.e-2,
                              device='cpu', dtype='float32', eps=1e-3, save_sample_path=False,
                              probability_flow=False, continuous=True, data_scalar=None):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            channel_modal=config.channel_modal)
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    # x_known = torch.from_numpy(x0).to(device).type(dtype_torch)
    y = torch.from_numpy(y).to(device).type(dtype_torch)
    # shape_sample = [len(y), config.num_channels, config.image_size, config.image_size]

    nf = config.num_frames
    ns = config.num_steps
    ncomp = config.num_components
    ol = config.overlap
    b = int(ns//(nf-ol)+1)      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [config.num_samples, b, nf, ncomp+config.num_modals-1, config.image_size]       # batch*b*nf*(c+npara)*h
    shape_sample = [config.num_samples, ns_real, ncomp+config.num_modals-1, config.image_size]     # batch*ns_real*(c+npara)*h

    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([config.num_samples, ns_real, ncomp+config.num_modals-1, config.image_size], dtype=dtype_torch, device=device)   # batch*ns_real*(c+npara)*h
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
        sample[:, :, ncomp:] = xx[:, 0, 0:1, ncomp:]
        return sample

    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()
    x_unknown = sde.prior_sampling(shape).to(device).float()    # batch*b*nf*(c+npara)*h
    
    x = x_unknown           # batch*b*(nf*c+npara)*h*w
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[])
    with tqdm(range(sde.N)) as tqdm_setting:
        for i in range(sde.N):
            t = timesteps[i]

            '''method 1 (batched)'''
            vec_t = torch.ones(shape[0]*b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h -> (b n) t c h')       # (batch*b)*nf*(c+npara)*h*w

            '''corrector'''
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)     # (batch*b)*nf*(c+npara)*h*w

            '''predictor'''
            z = torch.randn_like(x)
            zb = rearrange(z, 'b n t c h -> (b n) t c h')

            with torch.enable_grad():
                inp = temp.clone()                  # (batch*b)*nf*(c+npara)*h*w
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)          # (batch*b)*nf*(c+npara)*h*w
                with torch.no_grad():
                    f, G = sde.discretize(temp, vec_t)
                    rev_f = f - G[:, None, None, None] ** 2 * score.detach() * (0.5 if probability_flow else 1.)
                    rev_G = torch.zeros_like(G) if probability_flow else G
                    temp_mean = temp - rev_f
                    temp_u = temp_mean + rev_G[:, None, None, None] * zb
                # dps loss
                _, std = sde.marginal_prob(xb, vec_t)
                if isinstance(sde, VPSDE):
                    x0_hat = rearrange(std[:, None, None, None] ** 2 * score + inp, '(b n) t c h -> b n t c h', n=b)     # batch*b*nf*(c+npara)*h
                else:
                    alpha_sqrt_ = (1-std**2).sqrt()[:, None, None, None]
                    x0_hat = rearrange((std[:, None, None, None] ** 2 * score + inp)/alpha_sqrt_, '(b n) t c h -> b n t c h', n=b)     # batch*b*nf*(c+npara)*h
                x0_hat_temp = x_to_sample(x0_hat)
                if save_sample_path:
                    x0_hats.append(x0_hat.detach().cpu().numpy())

                var = std_y**2 + gamma * std**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_temp)) ** 2 / var).reshape(x0_hat.shape[0], -1), dim=-1)  # /scalar.sqrt()  *std[None, :, None, None]
                loss_dps = torch.sum(loss_dps, dim=0)  # /loss_dps.detach().mean().sqrt()
                if std_y is not None:
                    loss_dps = loss_dps/2.
                # loss_dps = loss_dps/loss_dps.detach().sqrt()    # normalize

                if b == 1:
                    # 当b=1时，使用时间步内的连续性损失
                    x0_curr = x0_hat[:, 0, :(nf-1), :ncomp]  # 当前帧
                    x0_next = x0_hat[:, 0, 1:nf, :ncomp]     # 下一帧
                    loss_consis = torch.sum((x0_curr.detach() - x0_next)**2)  # 计算相邻帧之间的差异
                else:
                    # 原有的计算方式
                    loss_consis = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, :ncomp].detach()-x0_hat[:, 1:, :ol, :ncomp])**2).reshape(x0_hat.shape[0], x0_hat.shape[1]-1, -1), dim=-1)
                
                loss_consis = torch.sum(loss_consis)  # 对所有批次求和
                
                # loss_consis_para的计算也需要修改
                if b == 1:
                    # 对于参数部分的连续性损失
                    loss_consis_para = torch.sum((x0_hat[:, 0, 1:, ncomp:] - x0_hat[:, 0, :1, ncomp:].detach())**2)
                else:
                    # 原有的计算方式
                    loss_consis_para = torch.sum(((x0_hat[:, 1:, :, ncomp:]-x0_hat[:, 0:1, :, ncomp:].detach())**2).reshape(x0_hat.shape[0], -1), dim=-1)
                
                loss_consis_para = torch.sum(loss_consis_para, dim=0)

                if beta is not None:
                    loss_eq, _ = kse_residual(inp, nf, 0.5, data_scalar)       # x0_hat_temp, ns_real
                    # scalar2 = scalar2.detach()
                    loss_eq = loss_eq/loss_eq.detach().sqrt()
                    loss = alpha * loss_dps + beta(t) * loss_eq + gamma1 * loss_consis + gamma2 * loss_consis_para  # /loss_dps.detach().sqrt()  /scalar2.mean().sqrt()
                    tqdm_setting.set_description(f'loss total: {loss.item():.5e} | loss dps: {alpha * loss_dps.item():.5e} | loss eq: {beta(t) * loss_eq.item():.5e} | loss consis: {gamma1 * loss_consis.item():.5e}')
                    losses['loss'].append(loss.item())
                    losses['loss_eq'].append(loss_eq.item())
                    losses['loss_dps'].append(loss_dps.item())
                    losses['loss_consis'].append(loss_consis.item())
                    losses['loss_consis_para'].append(loss_consis_para.item())
                    assert (not torch.isnan(loss_eq))
                else:
                    loss = alpha * loss_dps + gamma1 * loss_consis + gamma2 * loss_consis_para
                    tqdm_setting.set_description(f'loss total: {loss.item():.5e} | loss dps: {alpha * loss_dps.item():.5e} | loss consis: {gamma1 * loss_consis.item():.5e}')
                    losses['loss'].append(loss.item())
                    losses['loss_dps'].append(loss_dps.item())
                    losses['loss_consis'].append(loss_consis.item())
                    losses['loss_consis_para'].append(loss_consis_para.item())
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                temp = temp_u - dx     # (batch*b)*(nf*c+npara)*h*w
            #     # x = x_u
            temp = temp.detach()

            x = rearrange(temp, '(b n) t c h w -> b n t c h w', n=b)
            x_mean = rearrange(temp_mean, '(b n) t c h w -> b n t c h w', n=b)
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            tqdm_setting.update(1)

            # 在关键计算后添加监控
            if i % 25 == 0:
                monitor_data_range(x, f"第{i}步")

    return x_to_sample(x_mean).detach().cpu().numpy(), x0_hats if save_sample_path else None, losses


def vor_cal(u, v, grid_num, x_range):
    dx = (x_range[1]-x_range[0])/grid_num
    vor = (v[:-1, 1:]-v[:-1, :-1])/dx-(u[1:, :-1]-u[:-1, :-1])/dx
    return vor


def vor_cal_batch(x, grid_num, x_range, reverse=False, method='diff_1st', is_stagger=True):
    # method: 'diff_1st', 'spectral'
    vor = []
    for v in x:
        vx, vy = (v[1], v[0]) if reverse else (v[0], v[1])
        if 'diff_1st' in method:
            vor.append(vor_cal(vx, vy, grid_num, x_range))
        elif 'spectral' in method:
            vor.append(vor_cal_spectral(vx, vy, is_stagger=is_stagger))
        else:
            raise NotImplementedError('No such method for vorticity calculation!')
    return np.array(vor)


def vor_cal_plus(u, v, grid_num, x_range):
    omega = np.zeros((grid_num, grid_num))
    dx = dy = (x_range[1] - x_range[0]) / grid_num
    for i in range(1, grid_num - 1):
        for j in range(1, grid_num - 1):
            dudy = (u[i, j + 1] - u[i, j - 1]) / (2 * dy)
            dvdx = (v[i + 1, j] - v[i - 1, j]) / (2 * dx)
            omega[i, j] = dvdx - dudy
    return omega


def vor_cal_spectral(u, v, is_stagger=True):
    if is_stagger:
        # for staggered grid arrangement, we interpolate velocities from cell faces to cell centres
        u = 0.5 * (u + np.roll(u, 1, axis=1))
        v = 0.5 * (v + np.roll(v, -1, axis=0))
    k_max = len(u)//2
    k = np.concatenate([np.arange(0, k_max, 1), np.arange(-k_max, 0, 1)])
    k_x, k_y = np.meshgrid(k, k)
    F_u = np.fft.fft2(u)
    F_v = np.fft.fft2(v)
    # F_ux = 1j * k_x * F_u
    F_uy = 1j * k_y * F_u
    F_vx = 1j * k_x * F_v
    # F_vy = 1j * k_y * F_v
    # ux = np.fft.ifft2(F_ux)
    uy = np.fft.irfft2(F_uy[..., :k_max+1])
    vx = np.fft.irfft2(F_vx[..., :k_max+1])
    # vy = np.fft.ifft2(F_vy)
    return vx - uy


def mask_gen(input_shape, mask_ratio=0.5, seed=None):
    m = np.ones(input_shape)

    indices = [np.arange(i) for i in input_shape]
    I = np.meshgrid(*indices, indexing='ij')
    indices = np.array([index.reshape(-1) for index in I]).transpose(1, 0)
    num_pixel = len(indices)
    if seed is None:
        i_indices = np.random.choice(num_pixel, int(mask_ratio*num_pixel), replace=False)
    else:
        rng = np.random.RandomState(seed)
        i_indices = rng.choice(num_pixel, int(mask_ratio * num_pixel), replace=False)
    indices = indices[i_indices]
    m[tuple(indices.transpose(1, 0))] = 0
    m = m.astype(bool)
    return m


def plot_field(fields, row, col, dpi=100, q_range=None, save_fig=None):
    figsize = (col, row)
    fig, axes = plt.subplots(row, col, tight_layout=True, figsize=figsize, dpi=dpi)
    fields = fields.reshape(row, col, *fields.shape[1:])
    for i in range(row):
        for j in range(col):
            field = fields[i, j]
            pc = axes[i, j].pcolormesh(field, cmap='RdBu_r')
            if q_range is not None:
                pc.set_clim(q_range)
            axes[i, j].axis('off')
            axes[i, j].set_aspect(1)
    plt.show()
    if save_fig is not None:
        fig.savefig('./results/'+save_fig)


def cal_water_attr(t):
    attr_table = dict(
        t=[10, 20, 30, 40],
        rho=[999.7, 998.2, 995.7, 992.2],
        lamb=[0.574, 0.599, 0.618, 0.635],
        cp=[4191., 4183., 4174., 4174],
        # alpha=[20.e-6, 21.4e-6, 22.9e-6, 24.3e-6],
        mu=[1.306e-3, 1.004e-3, 0.8015e-4, 0.6533e-4],
        nu=[1.306e-6, 1.006e-6, 0.8050e-6, 0.6590e-6],
        Pr=[9.52, 7.02, 5.42, 4.31],
                      )
    t_low, t_high = attr_table['t'][0], attr_table['t'][-1]
    if t < t_low or t > t_high:
        raise ValueError(f'Input temperature out of range! Expect the input in range {t_low} to {t_high}')
    xs = attr_table['t']
    attr_t = dict()
    for key in attr_table.keys():
        v = attr_table[key]
        f = interp1d(xs, v, kind='linear')
        attr_t[key] = f(t)
    return attr_t


def sample_to_hot_wire(sample, coords, spacing, offsets=np.array([0, 0]), num_frame=10,
                       scalar=None, is_avg=True, use_para=False, weight=1.):
    # sample: b*(t*c+2)*h*w; coords: N*dim, N-points measurements of velocity; spacing: grid spacing
    device = sample.device
    indices = ((coords+offsets)/spacing).astype('int')

    if len(sample.shape) > 4:
        para = sample[:, :, num_frame:]
        sample = sample[:, :, :num_frame]
    else:
        para = sample[:, num_frame:num_frame+1]
        sample = sample[:, :num_frame]

    if scalar is not None:
        # scalar_std = torch.ones([1, len(sample[0]), 1, 1]).to(device)
        # scalar_mean = torch.zeros([1, len(sample[0]), 1, 1]).to(device)
        # scalar_std[:, :num_frame] = scalar_std[:, :num_frame]*scalar.std
        # scalar_mean[:, :num_frame] = scalar_mean[:, :num_frame]+scalar.mean
        # sample = sample*scalar.std+scalar.mean
        sample = scalar(sample)

    if len(sample.shape) > 4:
        obs = sample[:, :, :, indices[:, 0], indices[:, 1]]
        obs = torch.sqrt(obs[:, :, 0]**2+obs[:, :, 1]**2)
    else:
        obs = sample[:, :, indices[:, 0], indices[:, 1]]
        obs = torch.sqrt(obs[:, ::2]**2+obs[:, 1::2]**2)
    if is_avg:
        obs = obs.mean(1)
    if use_para:
        # obs = torch.cat([obs.reshape(len(obs), -1), weight*para.reshape(len(para), -1).mean(1)[:, None]], dim=-1)
        obs = torch.cat([obs.reshape(len(obs), -1), weight*para.reshape(len(para), -1)], dim=-1)

    return obs


def cal_rmse(gt, pred, normalize=True, reduct='sum'):
    # reduct = 'sum' or 'mean' etc.
    lib_name = np if isinstance(gt[0], np.ndarray) else torch
    reduct_fn = getattr(lib_name, reduct)
    rmse = []
    for a, b in zip(gt, pred):
        if normalize:
            coeff = 1./lib_name.sqrt(reduct_fn(a**2))
        else:
            coeff = 1.
        rmse.append(coeff*lib_name.sqrt(reduct_fn((a-b)**2)))
    return np.array(rmse) if isinstance(a, np.ndarray) else rmse


def cal_correlation(gt, pred, standardize=True, reduct='sum'):
    # standardize: whether to substract mean value of input data
    lib_name = np if isinstance(gt[0], np.ndarray) else torch
    reduct_fn = getattr(lib_name, reduct)
    cossim = []
    for a, b in zip(gt, pred):
        if standardize:
            a_mean = lib_name.mean(a)
            b_mean = lib_name.mean(b)
        else:
            a_mean = 0.
            b_mean = 0.
        a_norm = lib_name.sqrt(reduct_fn(a**2))
        b_norm = lib_name.sqrt(reduct_fn(b**2))
        cossim.append(reduct_fn((a-a_mean).reshape(-1)*(b-b_mean).reshape(-1))/(a_norm*b_norm))
    return np.array(cossim) if isinstance(a, np.ndarray) else cossim


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
