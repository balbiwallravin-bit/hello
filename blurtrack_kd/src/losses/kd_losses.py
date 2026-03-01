"""Knowledge distillation losses."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def normalize_heatmap(h: Tensor, gamma: float = 2.0, eps: float = 1e-6) -> Tensor:
    p = torch.pow(torch.clamp(h, 0.0, 1.0) + eps, gamma)
    return p / p.sum(dim=(2, 3), keepdim=True).clamp_min(eps)


class KDLoss(nn.Module):
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 0.5, d: float = 0.2, gamma: float = 2.0) -> None:
        super().__init__()
        self.a, self.b, self.c, self.d = a, b, c, d
        self.gamma = gamma
        gx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        gy = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer("sobel_x", gx)
        self.register_buffer("sobel_y", gy)
        self.l1 = nn.SmoothL1Loss(reduction="none")

    def _grad(self, h: Tensor) -> Tensor:
        gx = torch.nn.functional.conv2d(h, self.sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(h, self.sobel_y, padding=1)
        return torch.cat([gx, gy], dim=1)

    def forward(self, student_prob: Tensor, teacher_h: Tensor, mu_s: Tensor, teacher_m: dict[str, Tensor], conf_weight: Tensor) -> dict[str, Tensor]:
        p_s = normalize_heatmap(student_prob, self.gamma)
        p_t = normalize_heatmap(teacher_h, self.gamma)
        kl = (p_t * (torch.log(p_t + 1e-6) - torch.log(p_s + 1e-6))).sum(dim=(1, 2, 3))

        mu_t = teacher_m["mu_xy"]
        l_t = teacher_m["l"]
        sigma_t = torch.stack([teacher_m["sxx"], teacher_m["sxy"], teacher_m["syy"]], dim=1)
        sigma_s = torch.stack([teacher_m["sxx_s"], teacher_m["sxy_s"], teacher_m["syy_s"]], dim=1)
        l_mu = self.l1(mu_s, mu_t).mean(dim=1)
        w_sigma = torch.clamp((l_t.squeeze(1) - 3.0) / 5.0, 0.0, 1.0)
        l_sigma = self.l1(sigma_s, sigma_t).mean(dim=1)

        grad_l = torch.abs(self._grad(student_prob) - self._grad(teacher_h)).mean(dim=(1, 2, 3))
        w = conf_weight.squeeze(1)
        total = w * (self.a * kl + self.b * l_mu + self.c * w_sigma * l_sigma + self.d * grad_l)
        return {
            "loss": total.mean(),
            "kl": kl.mean(),
            "mu": l_mu.mean(),
            "sigma": l_sigma.mean(),
            "grad": grad_l.mean(),
            "mu_abs": torch.abs(mu_s - mu_t).mean(),
        }
