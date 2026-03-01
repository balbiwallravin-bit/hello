"""Pseudo label storage backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class PseudoStore:
    def __init__(self, root: str | Path, backend: str = "npz") -> None:
        self.root = Path(root)
        self.backend = backend
        self.hm_dir = self.root / "heatmaps"
        self.hm_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "meta.json"

    def write(self, sample_id: str, h4: np.ndarray, moments: dict[str, Any]) -> None:
        if self.backend == "npz":
            out = self.hm_dir / f"{sample_id}.npz"
            np.savez_compressed(out, H4=h4.astype(np.float16), **moments)
        elif self.backend == "zarr":
            try:
                import zarr
            except Exception as exc:
                raise RuntimeError("zarr backend requested but zarr not installed") from exc
            z = zarr.open(str(self.hm_dir / "heatmaps.zarr"), mode="a")
            z.create_dataset(sample_id, data=h4.astype(np.float16), overwrite=True)
            np.save(self.hm_dir / f"{sample_id}_moments.npy", moments, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def read(self, sample_id: str) -> dict[str, Any]:
        if self.backend == "npz":
            d = np.load(self.hm_dir / f"{sample_id}.npz")
            return {k: d[k] for k in d.files}
        if self.backend == "zarr":
            import zarr

            z = zarr.open(str(self.hm_dir / "heatmaps.zarr"), mode="r")
            m = np.load(self.hm_dir / f"{sample_id}_moments.npy", allow_pickle=True).item()
            return {"H4": np.array(z[sample_id]), **m}
        raise ValueError(f"Unsupported backend: {self.backend}")

    def write_meta(self, meta: dict[str, Any]) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
