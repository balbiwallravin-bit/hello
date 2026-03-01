from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.models.student_net import StudentExportWrapper, StudentRepUNetS


def test_export_onnx(tmp_path: Path) -> None:
    model = StudentRepUNetS()
    model.eval()
    wrap = StudentExportWrapper(model).eval()
    x = torch.randn(1, 9, 288, 512)
    pt = wrap(x)
    out_path = tmp_path / "student.onnx"
    torch.onnx.export(wrap, x, str(out_path), input_names=["input"], output_names=["mu_xy", "dir_xy", "l", "conf"], opset_version=17)

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": x.numpy().astype(np.float32)})
    assert ort_out[0].shape == tuple(pt[0].shape)
    for y in ort_out:
        assert np.isfinite(y).all()
