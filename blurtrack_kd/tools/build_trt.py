"""Build TensorRT engine from ONNX."""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--fp16", type=int, default=1)
    args = p.parse_args()
    try:
        import tensorrt as trt
    except Exception:
        print("TensorRT not installed. Skip building engine.")
        return
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("Failed to parse ONNX")
    config = builder.create_builder_config()
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_serialized_network(network, config)
    with open(args.engine, "wb") as f:
        f.write(engine)
    print(f"Saved engine: {args.engine}")


if __name__ == "__main__":
    main()
