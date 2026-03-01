"""Training entrypoint for student KD."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.frame_window_dataset import FrameWindowDataset
from src.losses.kd_losses import KDLoss
from src.models.student_net import StudentRepUNetS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FrameWindowDataset(cfg["data"]["index"], cfg["data"]["frames_root"], cfg["data"]["pseudo_dir"], augment=cfg["data"].get("augment", False))
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0)

    model = StudentRepUNetS(**cfg["model"]).to(device)
    loss_fn = KDLoss(**cfg["loss"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("amp", True) and device.type == "cuda")

    out_dir = Path(cfg["train"].get("out_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 1e9
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        losses = []
        for batch in dl:
            x = batch["x"].to(device)
            t_h = batch["teacher_H4"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(x)
                dec = model.decoder(out.heatmap_prob)
                teacher_m = {
                    "mu_xy": batch["teacher_mu_xy"].to(device),
                    "sxx": batch["teacher_sxx"].to(device),
                    "sxy": batch["teacher_sxy"].to(device),
                    "syy": batch["teacher_syy"].to(device),
                    "l": batch["teacher_l"].to(device),
                    "sxx_s": dec.sxx,
                    "sxy_s": dec.sxy,
                    "syy_s": dec.syy,
                }
                ld = loss_fn(out.heatmap_prob, t_h, out.mu_xy, teacher_m, batch["weight"].to(device))
                loss = ld["loss"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append({k: float(v.detach().cpu()) for k, v in ld.items()})
        avg = {k: sum(d[k] for d in losses) / len(losses) for k in losses[0]}
        print(f"epoch={epoch} loss={avg['loss']:.4f} kl={avg['kl']:.4f} mu={avg['mu']:.4f} sigma={avg['sigma']:.4f} grad={avg['grad']:.4f} mu_abs={avg['mu_abs']:.4f}")
        if avg["loss"] < best:
            best = avg["loss"]
            torch.save({"model": model.state_dict(), "config": cfg}, out_dir / "best.pt")


if __name__ == "__main__":
    main()
