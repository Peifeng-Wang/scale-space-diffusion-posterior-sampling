from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor, nn

import numpy as np

from src.optim.multiscale_schedule import MultiScaleSchedule
from src.physics.level_transfer import MultiLevelTransfer
from src.utils.lanczos import sample_from_simplified_sigma_batched


class FBPSampler:
    """Lightweight wrapper for FBP reconstruction used as a comparison baseline."""

    @torch.no_grad()
    def sample(self, y, physics, clamp_min: float = 0.0, clamp_max: float | None = 0.04825):
        x = physics.A_dagger(y)
        if clamp_max is None:
            x = x.clamp_min(clamp_min)
        else:
            x = x.clamp(clamp_min, clamp_max)
        return x.detach(), None


class DDPMSampler:
    """
    Implements the forward and reverse diffusion process for DDPM-style models.
    Handles noise addition during training (q_sample) and sampling (p_sample_loop).
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: str = 'cuda', clip_denoised: bool = True):
        """
        Args:
            num_timesteps: Total number of diffusion steps (T).
            beta_start: Starting value of the noise schedule.
            beta_end: Final value of the noise schedule.
            device: Device to store tensors on.
            clip_denoised: Whether to clip predicted x_0 to [-1, 1] during sampling.
        """
        self.device = device
        self.num_timesteps = num_timesteps
        self.clip_denoised = clip_denoised

        # Linear noise schedule (can replace with cosine schedule if needed)
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)

        # Register buffers
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # Precompute useful terms
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_recipm1_alphas = torch.sqrt(1.0 / alphas - 1)

        # Posterior mean coefficients
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    @torch.no_grad()
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion (add noise): q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, predict_x0: bool, zeta: float = 0.0) -> tuple:
        """
        Reverse diffusion (denoising) step:
            p(x_{t-1} | x_t) = N(mean, variance)
        The model predicts the added noise epsilon.
        """
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        alpha_t = self._extract(self.alphas, t, x_t.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)

        # Predict noise (epsilon)
        out = model(x_t, t) #[:, :1, ...]

        # Compute predicted x_0 (denoised image)
        if predict_x0:
            x0_pred = out
            eps_theta = (x_t - torch.sqrt(alpha_cumprod_t)  * x0_pred) / sqrt_one_minus_alphas_cumprod_t
        else:
            eps_theta = out
            x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * eps_theta) / torch.sqrt(alpha_cumprod_t)

        # Optional clipping
        if self.clip_denoised:
            x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

        # Fetch alpha_cumprod for the previous sub-sampled timestep
        alpha_cumprod_prev_t = torch.where(
            t_prev >= 0, 
            self.alphas_cumprod.gather(-1, torch.clamp(t_prev, min=0)), 
            torch.tensor(1.0, device=self.device)
        )
        alpha_cumprod_prev_t = alpha_cumprod_prev_t.reshape(t_prev.shape[0], *(((1,) * (len(x_t.shape) - 1))))

        # Combined directional noise term corresponding to paper Eq (15)
        noise = torch.randn_like(x_t) if (t_prev >= 0).any() else 0.0
        direction_eps = torch.sqrt(torch.tensor(1.0 - zeta, device=self.device)) * eps_theta + \
                        torch.sqrt(torch.tensor(zeta, device=self.device)) * noise

        # Compute x_{t-1} using DDIM deterministic / non-Markovian update formulation
        x_prev = torch.sqrt(alpha_cumprod_prev_t) * x0_pred + torch.sqrt(1.0 - alpha_cumprod_prev_t) * direction_eps

        return x_prev, x0_pred

    @torch.no_grad()
    def p_sample_loop(self, model, predict_x0: bool, shape: tuple, num_ddim_steps: int = 100, zeta: float = 0.0) -> torch.Tensor:
        """
        Iteratively sample from p(x_{t-1} | x_t) starting from x_T ~ N(0, I)
        """
        x = torch.randn(shape, device=self.device)

        batch_size = shape[0]

        # Generate a quadratic subsequence of timesteps as suggested in Section 3.5 of the paper
        times = np.linspace(0, np.sqrt(self.num_timesteps - 1), num_ddim_steps) ** 2
        times = np.round(times).astype(int)
        times = list(reversed(times))

        # Perform accelerated reverse diffusion sampling loop
        for i in range(len(times)):
            t_val = times[i]
            t_prev_val = times[i + 1] if i + 1 < len(times) else -1

            t = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
            t_prev = torch.full((batch_size,), t_prev_val, device=self.device, dtype=torch.long)

            x, x0 = self.p_sample(model, x, t, t_prev, predict_x0, zeta=zeta)
            
        return x

    @torch.no_grad()
    def sample(self, model, predict_x0: bool = False, num_ddim_steps: int = 100, zeta: float = 0.0,
               image_size: int = 256, batch_size: int = 1,
               channels: int = 1) -> torch.Tensor:
        """
        High-level sampling function.
        Generates new images from random noise.

        Args:
            model: Noise prediction model (predicts ε from x_t and t)
            image_size: Output image height/width
            batch_size: Number of samples to generate
            channels: Number of image channels
        """
        shape = (batch_size, channels, image_size, image_size)
        return self.p_sample_loop(model, predict_x0, shape, num_ddim_steps=num_ddim_steps, zeta=zeta)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Extract values from 1D tensor `a` at indices `t` and reshape for broadcasting.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *(((1,) * (len(x_shape) - 1))))

    def __repr__(self):
        return (f"Diffusion(num_timesteps={self.num_timesteps}, "
                f"beta_start={self.betas[0]:.1e}, beta_end={self.betas[-1]:.1e}, "
                f"clip_denoised={self.clip_denoised})")



class DiffPIRSampler(DDPMSampler):
    def __init__(self, *args, lambda_=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ = lambda_
        self.data_fidelity = None
        self.model = None

    def _predict_x0(self, x_t, t, predict_x0=False):
        out = self.model(x_t, t)

        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        if predict_x0:
            x0_pred = out
        else:
            eps_theta = out
            x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * eps_theta) / torch.sqrt(alpha_cumprod_t)

        if self.clip_denoised:
            x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

        return x0_pred

    def _gamma(self, i):
        abar = self.alphas_cumprod[i]
        return ((1.0 - abar) / abar) / self.lambda_

    def _posterior_step(self, x_t, x0_pred, t):
        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().reshape(x_t.shape[0], *([1] * (x_t.ndim - 1)))

        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(
        self,
        y,
        physics,
        data_fidelity,
        lambda_,
        model,
        predict_x0=False,
        x_init=None,
        t_start=None,
        noise=None,
        num_ddim_steps: int = 100,
        zeta: float = 0.0,
        record_history: bool = True,
    ):
        self.model = model
        self.lambda_ = lambda_
        self.data_fidelity = data_fidelity

        device = y.device
        b = y.shape[0]
        T = self.betas.shape[0]

        if t_start is None:
            t_start = T - 1

        if x_init is None:
            if hasattr(physics, "A_adjoint"):
                x = torch.randn_like(physics.A_adjoint(y))
            else:
                raise ValueError("Provide x_init, or define how to initialize x_T.")
        else:
            t = torch.full((b,), t_start, device=device, dtype=torch.long)
            if noise is None:
                noise = torch.randn_like(x_init)
            x = self.q_sample(x_init, t, noise=noise)

        times = np.linspace(0, np.sqrt(t_start), num_ddim_steps) ** 2
        times = np.round(times).astype(int)
        times = list(reversed(times))

        history_x0 = [] if record_history else None
        self.history_x0 = history_x0

        for idx in range(len(times)):
            i = times[idx]
            i_prev = times[idx + 1] if idx + 1 < len(times) else -1

            t = torch.full((b,), i, device=device, dtype=torch.long)
            t_prev = torch.full((b,), i_prev, device=device, dtype=torch.long)

            # same x0 prediction logic as DDPM
            x0 = self._predict_x0(x, t, predict_x0=predict_x0)

            # DiffPIR correction

            # need to normalize and denormalize
            x0_denorm =  0.04825 * x0

            x0_tilde_denorm = self.data_fidelity.prox(x0_denorm, y, physics, gamma=self._gamma(i))
        
            x0 = x0_tilde_denorm / 0.04825   

            # exact same posterior step as DDPM
            x = self.p_sample(lambda _x, _t: x0, x, t, t_prev, predict_x0=True, zeta=zeta)[0]

            if record_history:
                history_x0.append((i, x0.detach().cpu()))

        return x, history_x0


class DDSSampler(DiffPIRSampler):
    def __init__(self, *args, prox_gamma: float = 1e16, prox_max_iter: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.prox_gamma = prox_gamma
        self.prox_max_iter = prox_max_iter

    @torch.no_grad()
    def sample(
        self,
        y,
        physics,
        data_fidelity,
        lambda_,
        model,
        predict_x0=False,
        x_init=None,
        t_start=None,
        noise=None,
        num_ddim_steps: int = 100,
        zeta: float = 0.0,
        record_history: bool = True,
        prox_gamma: float | None = None,
        prox_max_iter: int | None = None,
    ):
        self.model = model
        self.lambda_ = lambda_
        self.data_fidelity = data_fidelity

        prox_gamma = self.prox_gamma if prox_gamma is None else prox_gamma
        prox_max_iter = self.prox_max_iter if prox_max_iter is None else prox_max_iter

        device = y.device
        b = y.shape[0]
        T = self.betas.shape[0]

        if t_start is None:
            t_start = T - 1

        if x_init is None:
            if hasattr(physics, "A_adjoint"):
                x = torch.randn_like(physics.A_adjoint(y))
            else:
                raise ValueError("Provide x_init, or define how to initialize x_T.")
        else:
            t = torch.full((b,), t_start, device=device, dtype=torch.long)
            if noise is None:
                noise = torch.randn_like(x_init)
            x = self.q_sample(x_init, t, noise=noise)

        times = np.linspace(0, np.sqrt(t_start), num_ddim_steps) ** 2
        times = np.round(times).astype(int)
        times = list(reversed(times))

        history_x0 = [] if record_history else None
        self.history_x0 = history_x0

        for idx in range(len(times)):
            i = times[idx]
            i_prev = times[idx + 1] if idx + 1 < len(times) else -1

            t = torch.full((b,), i, device=device, dtype=torch.long)
            t_prev = torch.full((b,), i_prev, device=device, dtype=torch.long)

            x0 = self._predict_x0(x, t, predict_x0=predict_x0)
            x0_denorm = 0.04825 * x0

            x0_tilde_denorm = physics.prox_l2(
                x0_denorm,
                y,
                gamma=prox_gamma,
                max_iter=prox_max_iter,
            )

            x0 = x0_tilde_denorm / 0.04825
            x = self.p_sample(lambda _x, _t: x0, x, t, t_prev, predict_x0=True, zeta=zeta)[0]

            if record_history:
                history_x0.append((i, x0.detach().cpu()))

        return x, history_x0



class Multi_Scale_DiffPIRSampler(DDPMSampler):
    def __init__(self, *args, lambda_=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ = lambda_
        self.data_fidelity = None
        self.model = None

    def _predict_x0(self, x_t, t, predict_x0=False):
        out = self.model(x_t, t)

        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        if predict_x0:
            x0_pred = out
        else:
            eps_theta = out
            x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * eps_theta) / torch.sqrt(alpha_cumprod_t)

        if self.clip_denoised:
            x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

        return x0_pred

    def _gamma(self, i):
        abar = self.alphas_cumprod[i]
        return ((1.0 - abar) / abar) / self.lambda_

    def _posterior_step(self, x_t, x0_pred, t):
        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().reshape(x_t.shape[0], *([1] * (x_t.ndim - 1)))

        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    def _transition_step(
            self,
            x,
            x0,
            t,
            t_prev,
            U,
            R,
            RT,
            M,
            MT,
    ):
        # U is the image-space lift. RT is the strict adjoint of R and is used
        # only through MT; using RT directly here would rescale image intensity.
        x0_prev = U(x0).clamp(0.0, 1.0)

        # abar_prev = self._extract(self.alphas_cumprod_prev, t, x0_prev.shape)
        # Fetch alpha_cumprod for arbitrary t_prev step to support DDIM jumps
        abar_prev = torch.where(
            t_prev >= 0,
            self.alphas_cumprod.gather(-1, torch.clamp(t_prev, min=0)),
            torch.tensor(1.0, device=self.device)
        )
        abar_prev = abar_prev.reshape(t_prev.shape[0], *(((1,) * (len(x0_prev.shape) - 1))))
        abar_t = self._extract(self.alphas_cumprod, t, x0_prev.shape)          # \bar{alpha}_t
        sigma2_prev = 1.0 - abar_prev
        sigma2_t = 1.0 - abar_t

        mu_prev = torch.sqrt(abar_prev) * x0_prev  # approx μ_{t-1}

        resid = x - M(mu_prev)  # in level_t space
        mu_post = mu_prev + (sigma2_prev / sigma2_t) * MT(resid)  # back to level_{t-1}

        # Lanczos currently supports batch_size=1 only, so we need to loop over the batch dimension if batch_size > 1.
        eta_posts = []

        for bi in range(x.shape[0]):
            x0_prev_i = x0_prev[bi:bi+1]
            s_t_i = torch.sqrt(self._extract(self.alphas, t[bi:bi+1], x0_prev_i.shape))

            def M_i(z):
                return s_t_i * R(z)

            def MT_i(z):
                return s_t_i * RT(z)

            eta_i = sample_from_simplified_sigma_batched(
                M_apply=M_i,
                MT_apply=MT_i,
                sigma_s=torch.sqrt(sigma2_prev[bi].flatten()[0]).item(),
                sigma_t=torch.sqrt(sigma2_t[bi].flatten()[0]).item(),
                hi_shape=x0_prev_i.shape,
                xi=None,
                lanczos_iters=40,
                estimate_lmax_iters=20,
                safety_eps=1e-3,
                device=x.device,
                dtype=x.dtype,
            )
            eta_posts.append(eta_i)

        eta_post = torch.cat(eta_posts, dim=0)

        return mu_post + eta_post
    

    @torch.no_grad()
    def sample(
        self,
        y,
        physics,
        data_fidelity,
        lambda_,
        model,
        levels,
        transition_ts,
        predict_x0=False,
        x_init=None,
        t_start=None,
        noise=None,
        num_ddim_steps: int = 100,
        zeta: float = 0.0,
        record_history: bool = True,
    ):
        self.lambda_ = lambda_
        self.data_fidelity = data_fidelity

        def measurement_at(level: int):
            if hasattr(y, "get_measurement"):
                return y.get_measurement(level)
            return y

        device = y.device
        b = y.shape[0]
        T = self.betas.shape[0]

        if t_start is None:
            t_start = T - 1
        
        # Create a multi-scale schedule based on the provided levels and transition time steps
        schedule = MultiScaleSchedule.from_block_spec(T=T, levels=levels, transition_ts=transition_ts)

        # base shape for level0
        if x_init is not None:
            base_shape = x_init.shape[1:]
        elif hasattr(physics, "get_physics"):
            base_shape = physics.get_physics(0).A_adjoint(measurement_at(0)).shape[1:]
        else:
            base_shape = physics.A_adjoint(measurement_at(0)).shape[1:]
        
        transfer = MultiLevelTransfer(n_level=max(levels)+1, init_shape=base_shape, device=device)

        # decide which level we start from
        level_start = schedule._level_of_t[t_start]

        def select_model(level: int):
            if isinstance(model, (list, tuple, nn.ModuleList)):
                return model[level] # level0 is the highest resolution model
            return model            # single model case

        # init x at the correct level
        if x_init is not None:
            x_init_lvl = transfer.projection_between_levels(x_init, n_from=0, n_to=level_start)
            t = torch.full((b,), t_start, device=device, dtype=torch.long)

            if noise is None:
                noise = torch.randn_like(x_init_lvl)
            else:
                noise = transfer.projection_between_levels(noise, n_from=0, n_to=level_start)

            x = self.q_sample(x_init_lvl, t, noise=noise)
        else:
            if hasattr(physics, "get_physics"):
                physics_start = physics.get_physics(level_start)
            else:
                physics_start = physics
            x = torch.randn_like(physics_start.A_adjoint(measurement_at(level_start)))

        times = np.linspace(0, np.sqrt(t_start), num_ddim_steps) ** 2
        times = np.round(times).astype(int)

        for ts in transition_ts:
            if 0 <= ts <= t_start:
                times = np.append(times, ts)
            if 0 <= ts + 1 <= t_start:
                times = np.append(times, ts + 1)

        times = np.unique(times)
        times = list(reversed(times))

        history_x0 = [] if record_history else None
        self.history_x0 = history_x0

        for idx in range(len(times)):
            i = times[idx]
            i_prev = times[idx + 1] if idx + 1 < len(times) else -1
            t = torch.full((b,), i, device=device, dtype=torch.long)
            t_prev = torch.full((b,), i_prev, device=device, dtype=torch.long)
            
            level_t = schedule._level_of_t[i]
            level_t_prev = schedule._level_of_t[i_prev if i_prev >= 0 else 0]

            is_transition = (i_prev >= 0) and (level_t != level_t_prev)

            s_t = torch.sqrt(self._extract(self.alphas, t, x.shape))    # alphas is from DDPMSampler

            def U(z):
                return transfer.image_lift_between_levels(z, n_from=level_t, n_to=level_t_prev)

            def R(z):
                return transfer.restriction_between_levels(z, n_from=level_t_prev, n_to=level_t)

            def RT(z):
                return transfer.restriction_adjoint_between_levels(z, n_from=level_t, n_to=level_t_prev)
            
            def M(z):
                return s_t * (z if level_t_prev == level_t else R(z))
            
            def MT(z):
                return s_t * (z if level_t_prev == level_t else RT(z))
            
            # select the appropriate model for the current level
            self.model = select_model(level_t)

            # same x0 prediction logic as DDPM
            x0 = self._predict_x0(x, t, predict_x0=predict_x0)

            # DiffPIR correction

            # need to normalize and denormalize
            x0_denorm =  0.04825 * x0


            # Depending on the physics implementation, we might need to get the appropriate level physics for the current time step
            if hasattr(physics, "get_physics"):
                physics_t = physics.get_physics(level_t)
            else:
                physics_t = physics
            y_t = measurement_at(level_t)

            x0_tilde_denorm = self.data_fidelity.prox(x0_denorm, y_t, physics_t, gamma=self._gamma(i))

            x0 = x0_tilde_denorm / 0.04825
                

            # exact same posterior step as DDPM
            x = self.p_sample(lambda _x, _t: x0, x, t, t_prev, predict_x0=True, zeta=zeta)[0] if not is_transition else self._transition_step(x, x0, t, t_prev, U, R, RT, M, MT)

            if record_history:
                history_x0.append((i, x0.detach().cpu()))

        return x, history_x0
