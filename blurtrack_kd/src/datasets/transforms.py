"""Simple multi-frame transforms."""

from __future__ import annotations

from typing import Callable

import numpy as np


class IdentityTransform:
    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return frames


class RandomBrightnessPerFrame:
    def __init__(self, enabled: bool = False, delta: float = 0.1) -> None:
        self.enabled = enabled
        self.delta = delta

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if not self.enabled:
            return frames
        out: list[np.ndarray] = []
        for f in frames:
            a = 1.0 + np.random.uniform(-self.delta, self.delta)
            b = np.random.uniform(-self.delta, self.delta)
            out.append(np.clip(f * a + b, 0.0, 1.0))
        return out


def build_transform(enabled: bool = False) -> Callable[[list[np.ndarray]], list[np.ndarray]]:
    return RandomBrightnessPerFrame(enabled=enabled)
