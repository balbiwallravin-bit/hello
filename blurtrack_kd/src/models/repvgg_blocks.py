"""RepVGG style blocks without depthwise conv."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.fc(self.pool(x))


class RepBlock(nn.Module):
    """Train-time multi-branch block; deploy-mode single 3x3 conv."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = False) -> None:
        super().__init__()
        self.conv3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False), nn.BatchNorm2d(out_ch))
        self.identity = nn.BatchNorm2d(in_ch) if stride == 1 and in_ch == out_ch else None
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        self.reparam: nn.Conv2d | None = None

    def forward(self, x: Tensor) -> Tensor:
        if self.reparam is not None:
            return self.act(self.se(self.reparam(x)))
        out = self.conv3(x) + self.conv1(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return self.act(self.se(out))

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
        w = conv.weight
        mean, var, gamma, beta, eps = bn.running_mean, bn.running_var, bn.weight, bn.bias, bn.eps
        std = torch.sqrt(var + eps)
        w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
        b_fused = beta - gamma * mean / std
        return w_fused, b_fused

    def _identity_kernel(self, channels: int, device: torch.device) -> Tensor:
        k = torch.zeros((channels, channels, 3, 3), device=device)
        for i in range(channels):
            k[i, i, 1, 1] = 1.0
        return k

    def switch_to_deploy(self) -> None:
        if self.reparam is not None:
            return
        w3, b3 = self._fuse_conv_bn(self.conv3[0], self.conv3[1])
        w1, b1 = self._fuse_conv_bn(self.conv1[0], self.conv1[1])
        w1_pad = torch.zeros_like(w3)
        w1_pad[:, :, 1:2, 1:2] = w1
        w = w3 + w1_pad
        b = b3 + b1
        if self.identity is not None:
            channels = w.shape[0]
            id_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=False).to(w.device)
            id_conv.weight.data = self._identity_kernel(channels, w.device)
            wi, bi = self._fuse_conv_bn(id_conv, self.identity)
            w = w + wi
            b = b + bi
        self.reparam = nn.Conv2d(self.conv3[0].in_channels, self.conv3[0].out_channels, 3, self.conv3[0].stride, 1, bias=True)
        self.reparam.weight.data = w
        self.reparam.bias.data = b
        for attr in ["conv3", "conv1", "identity"]:
            if hasattr(self, attr):
                delattr(self, attr)
