"""Export student model to ONNX."""

from __future__ import annotations

import argparse

import torch

from src.models.student_net import StudentExportWrapper, StudentRepUNetS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--static_shape", nargs=4, type=int, default=[1, 9, 288, 512])
    ap.add_argument("--export_params_only", type=int, default=1)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {}).get("model", {})
    model = StudentRepUNetS(**cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    model.switch_to_deploy()
    wrap = StudentExportWrapper(model).eval()
    dummy = torch.randn(*args.static_shape)
    torch.onnx.export(
        wrap,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["mu_xy", "dir_xy", "l", "conf"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"exported: {args.out}")


if __name__ == "__main__":
    main()
