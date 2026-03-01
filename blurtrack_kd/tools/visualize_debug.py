"""Debug visualization for teacher/student outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.datasets.frame_window_dataset import FrameWindowDataset
from src.models.student_net import StudentRepUNetS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames_root", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--pseudo_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = FrameWindowDataset(args.index, args.frames_root, args.pseudo_dir)
    model = StudentRepUNetS()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"], strict=False)
    model.eval()

    for i in range(min(args.n, len(ds))):
        b = ds[i]
        with torch.no_grad():
            o = model(b["x"].unsqueeze(0))
        frame = (b["x"][6:9].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.circle(frame, (int(o.mu_xy[0, 0]), int(o.mu_xy[0, 1])), 4, (255, 0, 0), 1)
        cv2.circle(frame, (int(b["teacher_mu_xy"][0]), int(b["teacher_mu_xy"][1])), 4, (0, 255, 0), 1)
        cv2.imwrite(str(out_dir / f"debug_{i:03d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
