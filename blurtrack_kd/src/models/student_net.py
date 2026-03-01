"""Student RepUNet model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .moment_decoder import MomentDecoder
from .repvgg_blocks import RepBlock


def _match_hw(x: Tensor, ref: Tensor) -> Tensor:
    """Match spatial size of x to ref by crop then pad."""
    h_ref, w_ref = ref.shape[-2], ref.shape[-1]
    h, w = x.shape[-2], x.shape[-1]
    if h > h_ref:
        x = x[..., :h_ref, :]
    if w > w_ref:
        x = x[..., :, :w_ref]
    if x.shape[-2] < h_ref or x.shape[-1] < w_ref:
        pad_h = h_ref - x.shape[-2]
        pad_w = w_ref - x.shape[-1]
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x


@dataclass
class StudentOutput:
    heatmap_logits: Tensor
    heatmap_prob: Tensor
    mu_xy: Tensor
    dir_xy: Tensor
    l: Tensor
    conf: Tensor


class StudentRepUNetS(nn.Module):
    def __init__(self, base_ch: int = 32, stride: int = 4, gamma: float = 2.0, k_len: float = 4.0) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(9, base_ch, 3, 2, 1, bias=False), nn.BatchNorm2d(base_ch), nn.SiLU(inplace=True))
        self.enc1 = nn.Sequential(RepBlock(base_ch, base_ch, 1, use_se=True), RepBlock(base_ch, base_ch * 2, 2, use_se=True))
        self.enc2 = nn.Sequential(RepBlock(base_ch * 2, base_ch * 2), RepBlock(base_ch * 2, base_ch * 4, 2))
        self.mid = nn.Sequential(RepBlock(base_ch * 4, base_ch * 4), RepBlock(base_ch * 4, base_ch * 4))
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, 1, 1),
            nn.SiLU(inplace=True),
        )
        self.dec1 = RepBlock(base_ch * 2, base_ch * 2)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
            nn.SiLU(inplace=True),
        )
        self.dec2 = RepBlock(base_ch, base_ch)
        self.head = nn.Conv2d(base_ch, 1, 1)
        self.decoder = MomentDecoder(h=72, w=128, stride=stride, gamma=gamma, k_len=k_len)

    def forward(self, x: Tensor) -> StudentOutput:
        s = self.stem(x)
        e1 = self.enc1(s)
        e2 = self.enc2(e1)
        m = self.mid(e2)
        up1 = _match_hw(self.up1(m), e1)
        d1 = self.dec1(up1 + e1)
        up2 = _match_hw(self.up2(d1), s)
        d2 = self.dec2(up2 + s)
        logits = self.head(d2)

        # Keep decoder input stable at 72x128 (stride=4 for 288x512 default).
        if logits.shape[-2:] != (72, 128):
            logits = F.interpolate(logits, size=(72, 128), mode="bilinear", align_corners=False)

        prob = torch.sigmoid(logits)
        mo = self.decoder(prob)
        return StudentOutput(logits, prob, mo.mu_xy_img, mo.dir_xy, mo.l_img, mo.conf)

    def switch_to_deploy(self) -> None:
        for m in self.modules():
            if isinstance(m, RepBlock):
                m.switch_to_deploy()


class StudentExportWrapper(nn.Module):
    """ONNX export wrapper that returns tensors only."""

    def __init__(self, model: StudentRepUNetS) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        o = self.model(x)
        return o.mu_xy, o.dir_xy, o.l, o.conf
