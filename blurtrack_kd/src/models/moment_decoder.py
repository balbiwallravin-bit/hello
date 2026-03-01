"""Differentiable moment decoder for heatmap-to-params conversion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MomentOutput:
    """Container for decoded moment outputs."""

    mu_xy_img: Tensor
    dir_xy: Tensor
    l_img: Tensor
    conf: Tensor
    sxx: Tensor
    sxy: Tensor
    syy: Tensor


class MomentDecoder(nn.Module):
    """Decode a heatmap into center, direction and half-length with tensor-only ops."""

    def __init__(
        self,
        h: int = 72,
        w: int = 128,
        stride: float = 4.0,
        gamma: float = 2.0,
        k_len: float = 4.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.h = h
        self.w = w
        self.stride = stride
        self.gamma = gamma
        self.k_len = k_len
        self.eps = eps
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
        self.register_buffer("grid_x", xx.view(1, 1, h, w), persistent=False)
        self.register_buffer("grid_y", yy.view(1, 1, h, w), persistent=False)

    def _normalize(self, heatmap_prob: Tensor) -> Tensor:
        h = torch.clamp(heatmap_prob, 0.0, 1.0)
        p = torch.pow(h + self.eps, self.gamma)
        z = p.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return p / z

    def forward(self, heatmap_prob: Tensor) -> MomentOutput:
        """Decode heatmap probabilities [B,1,H,W]."""
        p = self._normalize(heatmap_prob)
        mu_x = (p * self.grid_x).sum(dim=(2, 3))
        mu_y = (p * self.grid_y).sum(dim=(2, 3))
        mux = mu_x.view(-1, 1, 1, 1)
        muy = mu_y.view(-1, 1, 1, 1)

        dx = self.grid_x - mux
        dy = self.grid_y - muy
        sxx = (p * dx * dx).sum(dim=(2, 3))
        sxy = (p * dx * dy).sum(dim=(2, 3))
        syy = (p * dy * dy).sum(dim=(2, 3))

        t = sxx - syy
        u = torch.sqrt(t * t + 4.0 * sxy * sxy + self.eps)
        vx = 2.0 * sxy
        vy = syy - sxx + u
        vnorm = torch.sqrt(vx * vx + vy * vy + self.eps)
        dir_xy = torch.cat([vx / vnorm, vy / vnorm], dim=1)

        lam_max = 0.5 * (sxx + syy + u)
        sigma_major = torch.sqrt(lam_max + self.eps)

        mu_xy_img = torch.cat([(mu_x + 0.5) * self.stride, (mu_y + 0.5) * self.stride], dim=1)
        l_img = (self.k_len * sigma_major * self.stride).view(-1, 1)
        conf = heatmap_prob.mean(dim=(2, 3))

        return MomentOutput(mu_xy_img=mu_xy_img, dir_xy=dir_xy, l_img=l_img, conf=conf, sxx=sxx, sxy=sxy, syy=syy)
