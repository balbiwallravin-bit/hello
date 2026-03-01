"""Training entrypoint for student KD."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.datasets.frame_window_dataset import FrameWindowDataset
from src.losses.kd_losses import KDLoss
from src.models.student_net import StudentRepUNetS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if "train" not in cfg or "lr" not in cfg["train"]:
        raise KeyError("Config must contain train.lr")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FrameWindowDataset(
        cfg["data"]["index"],
        cfg["data"]["frames_root"],
        cfg["data"].get("pseudo_dir"),
        augment=cfg["data"].get("augment", False),
        cfg_data=cfg.get("data", {}),
    )
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )

    model_kwargs = dict(cfg.get("model", {}))
    msig = inspect.signature(StudentRepUNetS.__init__)
    macc = set(msig.parameters.keys()) - {"self"}
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in macc}
    model = StudentRepUNetS(**model_kwargs).to(device)

    loss_kwargs = dict(cfg.get("loss", {}))
    lsig = inspect.signature(KDLoss.__init__)
    lacc = set(lsig.parameters.keys()) - {"self"}
    loss_kwargs = {k: v for k, v in loss_kwargs.items() if k in lacc}
    loss_fn = KDLoss(**loss_kwargs).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scaler = GradScaler("cuda", enabled=cfg["train"].get("amp", True) and device.type == "cuda")

    out_dir = Path(cfg["train"].get("out_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 1e9
    log_every = int(cfg["train"].get("log_every", 50))
    save_best = bool(cfg["train"].get("save_best", True))

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        losses = []
        for step, batch in enumerate(dl):
            x = batch["x"].to(device)
            t_h = batch["teacher_H4"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=scaler.is_enabled()):
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

            if (step + 1) % log_every == 0:
                print(
                    f"epoch={epoch} step={step + 1}/{len(dl)} "
                    f"loss={losses[-1]['loss']:.4f} mu_abs={losses[-1]['mu_abs']:.4f}"
                )

        avg = {k: sum(d[k] for d in losses) / len(losses) for k in losses[0]}
        print(
            f"epoch={epoch} loss={avg['loss']:.4f} kl={avg['kl']:.4f} "
            f"mu={avg['mu']:.4f} sigma={avg['sigma']:.4f} grad={avg['grad']:.4f} mu_abs={avg['mu_abs']:.4f}"
        )
        if avg["loss"] < best:
            best = avg["loss"]
            if save_best:
                torch.save({"model": model.state_dict(), "config": cfg}, out_dir / "best.pt")


if __name__ == "__main__":
    main()
