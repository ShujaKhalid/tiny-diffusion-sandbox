import torch
import torch.nn as nn

import numpy as np
from dataclasses import dataclass

@dataclass(eq=False, repr=False)
class NoiseScheduler:
    def __init__(self, cfg):
        super().__init__()
        # compute parameters
        self.betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = nn.functional.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_inv_alphas = torch.sqrt(1 / self.alphas)

        # parameters for forward process
        self.coef_noise_mu = self.sqrt_alphas_cumprod
        self.coef_noise_sigma = self.sqrt_one_minus_alphas_cumprod

        # parameters for backward process
        self.coef_denoise_mu_1 = self.sqrt_inv_alphas
        self.coef_denoise_mu_2 = (
                self.betas / torch.sqrt(1 - self.alphas_cumprod) * self.sqrt_inv_alphas
        )
        self.coef_denoise_sigma = torch.sqrt(
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_0, t, eps_target):
        mu = self.coef_noise_mu[t] * x_0
        sigma = self.coef_noise_sigma[t]
        return mu + sigma * eps_target

    def remove_noise(self, x_t, eps_pred, t):
        mu = self.coef_denoise_mu_1[t] * x_t - self.coef_denoise_mu_2[t] * eps_pred
        sigma = self.coef_denoise_sigma[t]
        z = torch.randn_like(eps_pred)
        return mu + sigma * z


