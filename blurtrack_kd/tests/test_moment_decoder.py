from __future__ import annotations

import math

import numpy as np
import torch

from src.models.moment_decoder import MomentDecoder


def make_line_heatmap(h: int = 72, w: int = 128, cx: float = 64.0, cy: float = 36.0, angle_deg: float = 30.0, length: float = 20.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    s = (xx - cx) * dx + (yy - cy) * dy
    d = (xx - cx) * dy - (yy - cy) * dx
    hmap = np.exp(-0.5 * (d / 1.2) ** 2) * np.exp(-0.5 * (s / (length / 3)) ** 2)
    return hmap.astype(np.float32)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = abs(float((v1 * v2).sum()))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(math.acos(dot))


def test_moment_decoder_line() -> None:
    dec = MomentDecoder()
    h1 = make_line_heatmap(angle_deg=20, length=14)
    h2 = make_line_heatmap(angle_deg=20, length=30)
    t1 = torch.from_numpy(h1)[None, None]
    t2 = torch.from_numpy(h2)[None, None]
    o1 = dec(t1)
    o2 = dec(t2)

    mu = o1.mu_xy_img[0].numpy() / 4.0 - 0.5
    assert abs(mu[0] - 64.0) < 1.0
    assert abs(mu[1] - 36.0) < 1.0

    gt = np.array([math.cos(math.radians(20)), math.sin(math.radians(20))], dtype=np.float32)
    ang = angle_between(o1.dir_xy[0].numpy(), gt)
    assert ang < 5.0

    assert float(o2.l_img[0, 0]) > float(o1.l_img[0, 0])
