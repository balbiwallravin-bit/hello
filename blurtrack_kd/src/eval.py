"""Minimal evaluation script."""

from __future__ import annotations

import argparse

import torch

from src.datasets.frame_window_dataset import FrameWindowDataset
from src.models.student_net import StudentRepUNetS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--pseudo_dir", required=True)
    args = ap.parse_args()

    ds = FrameWindowDataset(args.index, args.frames_root, args.pseudo_dir)
    model = StudentRepUNetS()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(min(len(ds), 50)):
            b = ds[i]
            o = model(b["x"].unsqueeze(0))
            errs.append(torch.abs(o.mu_xy[0] - b["teacher_mu_xy"]).mean().item())
    print(f"avg_mu_err={sum(errs)/len(errs):.4f}")


if __name__ == "__main__":
    main()
