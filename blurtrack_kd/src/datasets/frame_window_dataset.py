"""3-frame window dataset and index builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .pseudo_store import PseudoStore
from .transforms import build_transform


def _to_rel_or_abs(path: Path, base: Path) -> str:
    """Return relative path when possible, otherwise absolute path string."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def build_index(frames_root: Path, out: Path, mode: str = "predict_middle", pseudo_root: Path | None = None) -> None:
    """Build index for either generic frame dirs or segment_id/frames_roi layout."""
    records: list[dict[str, Any]] = []
    pseudo_root = pseudo_root or (frames_root.parent / "pseudo" / "heatmaps")

    segments = sorted([p for p in frames_root.iterdir() if p.is_dir() and (p / "frames_roi").is_dir()])
    if segments:
        for seg_dir in segments:
            frame_dir = seg_dir / "frames_roi"
            frames = sorted(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png")))
            if len(frames) < 3:
                continue
            for i in range(1, len(frames) - 1):
                if mode != "predict_middle":
                    raise ValueError("frames_roi layout currently supports mode=predict_middle only")
                try:
                    t = int(frames[i].stem)
                except ValueError:
                    t = i
                records.append(
                    {
                        "sample_id": f"{seg_dir.name}_{t:05d}",
                        "segment_id": seg_dir.name,
                        "t": t,
                        "img_paths": [
                            _to_rel_or_abs(frames[i - 1], frames_root),
                            _to_rel_or_abs(frames[i], frames_root),
                            _to_rel_or_abs(frames[i + 1], frames_root),
                        ],
                        "label_npz": str(pseudo_root / seg_dir.name / f"{frames[i].stem}.npz"),
                    }
                )
    else:
        for vid_dir in sorted([p for p in frames_root.iterdir() if p.is_dir()]):
            frames = sorted(list(vid_dir.glob("*.jpg")) + list(vid_dir.glob("*.png")))
            for i in range(2, len(frames)):
                target_i = i - 1 if mode == "predict_middle" else i
                sid = f"{vid_dir.name}_{target_i:06d}"
                records.append(
                    {
                        "sample_id": sid,
                        "video_id": vid_dir.name,
                        "frames": [
                            str(frames[i - 2].relative_to(frames_root)),
                            str(frames[i - 1].relative_to(frames_root)),
                            str(frames[i].relative_to(frames_root)),
                        ],
                        "target_frame": str(frames[target_i].relative_to(frames_root)),
                    }
                )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class FrameWindowDataset(Dataset):
    def __init__(
        self,
        index_path: str,
        frames_root: str,
        pseudo_dir: str | None = None,
        augment: bool = False,
        cfg_data: dict[str, Any] | None = None,
    ) -> None:
        self.frames_root = Path(frames_root)
        self.cfg_data = cfg_data or {}
        self.items = [json.loads(line) for line in Path(index_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.store = PseudoStore(pseudo_dir, backend="npz") if pseudo_dir else None
        self.transform = build_transform(enabled=augment)

    def __len__(self) -> int:
        return len(self.items)

    def _read_img(self, p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h = int(self.cfg_data.get("input_h", 288))
        w = int(self.cfg_data.get("input_w", 512))
        interp = cv2.INTER_AREA if (img.shape[0] > h or img.shape[1] > w) else cv2.INTER_LINEAR
        img = cv2.resize(img, (w, h), interpolation=interp)
        return img.astype(np.float32) / 255.0

    def _resolve_img_path(self, p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else self.frames_root / pp

    @staticmethod
    def _read_label_npz(path: str) -> dict[str, Any]:
        d = np.load(path)
        if "H4" in d.files:
            h4 = d["H4"]
        elif "hm" in d.files:
            h4 = d["hm"]
        elif "heatmap" in d.files:
            h4 = d["heatmap"]
        else:
            h4 = d[d.files[0]]

        if h4.ndim == 2:
            h4 = h4[None, ...]

        return {
            "H4": h4.astype(np.float32),
            "mu_x_img": float(d["mu_x_img"]) if "mu_x_img" in d.files else 0.0,
            "mu_y_img": float(d["mu_y_img"]) if "mu_y_img" in d.files else 0.0,
            "sxx": float(d["sxx"]) if "sxx" in d.files else 0.0,
            "sxy": float(d["sxy"]) if "sxy" in d.files else 0.0,
            "syy": float(d["syy"]) if "syy" in d.files else 0.0,
            "l_img": float(d["l_img"]) if "l_img" in d.files else 0.0,
            "conf": float(d["conf"]) if "conf" in d.files else float(np.max(h4)),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx]
        frame_paths = item.get("img_paths", item.get("frames"))
        if frame_paths is None:
            raise KeyError("Index item missing both `img_paths` and `frames` fields")
        frames = [self._read_img(self._resolve_img_path(rel)) for rel in frame_paths]
        frames = self.transform(frames)
        x = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
        out: dict[str, torch.Tensor] = {"x": torch.from_numpy(x).float(), "sample_id": item.get("sample_id", str(idx))}

        if "label_npz" in item:
            p = self._read_label_npz(item["label_npz"])
        elif self.store is not None:
            p = self.store.read(item["sample_id"])
        else:
            return out

        out.update(
            {
                "teacher_H4": torch.from_numpy(p["H4"]).float(),
                "teacher_mu_xy": torch.tensor([p["mu_x_img"], p["mu_y_img"]], dtype=torch.float32),
                "teacher_sxx": torch.tensor([p["sxx"]], dtype=torch.float32),
                "teacher_sxy": torch.tensor([p["sxy"]], dtype=torch.float32),
                "teacher_syy": torch.tensor([p["syy"]], dtype=torch.float32),
                "teacher_l": torch.tensor([p["l_img"]], dtype=torch.float32),
                "weight": torch.tensor([p.get("conf", 1.0)], dtype=torch.float32),
            }
        )
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--build_index", action="store_true")
    p.add_argument("--frames_root", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--mode", type=str, default="predict_middle", choices=["predict_middle", "predict_last"])
    p.add_argument("--pseudo_root", type=str, default="")
    args = p.parse_args()
    if args.build_index:
        pseudo_root = Path(args.pseudo_root) if args.pseudo_root else None
        build_index(Path(args.frames_root), Path(args.out), mode=args.mode, pseudo_root=pseudo_root)


if __name__ == "__main__":
    main()
