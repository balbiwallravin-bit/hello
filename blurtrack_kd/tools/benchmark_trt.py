"""Benchmark TensorRT engine."""

from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--iters", type=int, default=2000)
    args = p.parse_args()
    try:
        import tensorrt as trt
    except Exception:
        print("TensorRT not installed. Skip benchmark.")
        return
    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Invalid engine")
    # lightweight placeholder benchmark
    tic = time.time()
    for _ in range(args.warmup + args.iters):
        _ = np.empty((1, 9, 288, 512), dtype=np.float16)
    dt = time.time() - tic
    print(f"placeholder_iter_time={(dt/(args.warmup+args.iters))*1000:.3f} ms")


if __name__ == "__main__":
    main()
