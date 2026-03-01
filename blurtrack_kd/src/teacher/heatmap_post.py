"""Teacher heatmap post-processing with differentiable moments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.models.moment_decoder import MomentDecoder


def adapt_teacher_heatmap(h: np.ndarray, out_h: int = 72, out_w: int = 128) -> np.ndarray:
    t = torch.from_numpy(h).float()
    if t.ndim == 3:
        t = t.unsqueeze(1)
    if t.shape[-2:] != (out_h, out_w):
        t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return t.numpy()


def compute_moments(h4: np.ndarray, stride: int = 4, gamma: float = 2.0, k_len: float = 4.0) -> dict[str, np.ndarray]:
    dec = MomentDecoder(h=h4.shape[-2], w=h4.shape[-1], stride=float(stride), gamma=gamma, k_len=k_len)
    with torch.no_grad():
        o = dec(torch.from_numpy(h4).float())
    return {
        "mu_x_img": o.mu_xy_img[:, 0].cpu().numpy(),
        "mu_y_img": o.mu_xy_img[:, 1].cpu().numpy(),
        "sxx": o.sxx[:, 0].cpu().numpy(),
        "sxy": o.sxy[:, 0].cpu().numpy(),
        "syy": o.syy[:, 0].cpu().numpy(),
        "l_img": o.l_img[:, 0].cpu().numpy(),
        "conf": o.conf[:, 0].cpu().numpy(),
    }
