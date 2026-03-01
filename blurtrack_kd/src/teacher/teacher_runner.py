"""Run teacher ONNX (or fake teacher) to create pseudo labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.datasets.pseudo_store import PseudoStore
from src.teacher.heatmap_post import adapt_teacher_heatmap, compute_moments


def load_frames(frames_root: Path, rels: list[str]) -> np.ndarray:
    arrs = []
    for r in rels:
        rp = Path(r)
        img_path = rp if rp.is_absolute() else frames_root / rp
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        arrs.append(img.transpose(2, 0, 1))
    return np.concatenate(arrs, axis=0)


def fake_teacher(batch_x: np.ndarray) -> np.ndarray:
    b = batch_x.shape[0]
    h, w = 72, 128
    out = np.zeros((b, 1, h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(b):
        cx = np.random.uniform(20, w - 20)
        cy = np.random.uniform(15, h - 15)
        ang = np.random.uniform(0, np.pi)
        dx, dy = np.cos(ang), np.sin(ang)
        d = (xx - cx) * dy - (yy - cy) * dx
        s = (xx - cx) * dx + (yy - cy) * dy
        out[i, 0] = np.exp(-0.5 * (d / 1.5) ** 2) * np.exp(-0.5 * (s / 8.0) ** 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--teacher_onnx", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--save_backend", default="npz", choices=["npz", "zarr"])
    ap.add_argument("--k_len", type=float, default=4.0)
    args = ap.parse_args()

    if args.teacher_onnx:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("onnxruntime is required for --teacher_onnx mode") from exc
        sess = ort.InferenceSession(args.teacher_onnx, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
    else:
        sess = None
        input_name = ""

    rows = [json.loads(x) for x in Path(args.index).read_text(encoding="utf-8").splitlines() if x.strip()]
    store = PseudoStore(args.out_dir, backend=args.save_backend)
    moments_rows = []

    for row in rows:
        frame_keys = row.get("img_paths", row.get("frames"))
        if frame_keys is None:
            raise KeyError("index row missing img_paths/frames")
        x = load_frames(Path(args.frames_root), frame_keys)[None, ...]
        if sess is not None:
            h = sess.run(None, {input_name: x.astype(np.float32)})[0]
            h4 = adapt_teacher_heatmap(h)
        else:
            h4 = fake_teacher(x)
        mm = compute_moments(h4, stride=args.stride, gamma=args.gamma, k_len=args.k_len)
        m_single = {k: float(v[0]) for k, v in mm.items()}
        store.write(row["sample_id"], h4[0], m_single)
        moments_rows.append({"sample_id": row["sample_id"], **m_single})

    pd.DataFrame(moments_rows).to_csv(Path(args.out_dir) / "moments.csv", index=False)
    store.write_meta({"stride": args.stride, "gamma": args.gamma, "version": "v1"})


if __name__ == "__main__":
    main()
