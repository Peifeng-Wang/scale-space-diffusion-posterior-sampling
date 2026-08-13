from __future__ import annotations

import numpy as np
import torch


class DDPMSampler:
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: str = "cuda", clip_denoised: bool = True):
        self.device = device
        self.num_timesteps = num_timesteps
        self.clip_denoised = clip_denoised

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_recipm1_alphas = torch.sqrt(1.0 / alphas - 1)

        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    @torch.no_grad()
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, predict_x0: bool, zeta: float = 0.0) -> tuple:
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)

        out = model(x_t, t)

        if predict_x0:
            x0_pred = out
            eps_theta = (x_t - torch.sqrt(alpha_cumprod_t) * x0_pred) / sqrt_one_minus_alphas_cumprod_t
        else:
            eps_theta = out
            x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * eps_theta) / torch.sqrt(alpha_cumprod_t)

        if self.clip_denoised:
            x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

        alpha_cumprod_prev_t = torch.where(
            t_prev >= 0,
            self.alphas_cumprod.gather(-1, torch.clamp(t_prev, min=0)),
            torch.tensor(1.0, device=self.device),
        )
        alpha_cumprod_prev_t = alpha_cumprod_prev_t.reshape(t_prev.shape[0], *(((1,) * (len(x_t.shape) - 1))))

        noise = torch.randn_like(x_t) if (t_prev >= 0).any() else 0.0
        direction_eps = torch.sqrt(torch.tensor(1.0 - zeta, device=self.device)) * eps_theta + \
                        torch.sqrt(torch.tensor(zeta, device=self.device)) * noise

        x_prev = torch.sqrt(alpha_cumprod_prev_t) * x0_pred + torch.sqrt(1.0 - alpha_cumprod_prev_t) * direction_eps
        return x_prev, x0_pred

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *(((1,) * (len(x_shape) - 1))))


class DDSSampler(DDPMSampler):
    def __init__(self, *args, lambda_=1.0, prox_gamma: float = 1e16, prox_max_iter: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ = lambda_
        self.prox_gamma = prox_gamma
        self.prox_max_iter = prox_max_iter
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
        record_history: bool = False,
        prox_gamma: float | None = None,
        prox_max_iter: int | None = None,
    ):
        self.model = model
        self.lambda_ = lambda_
        self.data_fidelity = data_fidelity

        prox_gamma = self.prox_gamma if prox_gamma is None else prox_gamma
        prox_max_iter = self.prox_max_iter if prox_max_iter is None else prox_max_iter

        device = y.device
        batch_size = y.shape[0]
        T = self.betas.shape[0]

        if t_start is None:
            t_start = T - 1

        if x_init is None:
            if hasattr(physics, "A_adjoint"):
                x = torch.randn_like(physics.A_adjoint(y))
            else:
                raise ValueError("Provide x_init, or define how to initialize x_T.")
        else:
            t = torch.full((batch_size,), t_start, device=device, dtype=torch.long)
            if noise is None:
                noise = torch.randn_like(x_init)
            x = self.q_sample(x_init, t, noise=noise)

        times = np.linspace(0, np.sqrt(t_start), num_ddim_steps) ** 2
        times = np.round(times).astype(int)
        times = list(reversed(times))

        history_x0 = [] if record_history else None

        for i, i_prev in zip(times, times[1:] + [-1]):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_prev = torch.full((batch_size,), i_prev, device=device, dtype=torch.long)

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
