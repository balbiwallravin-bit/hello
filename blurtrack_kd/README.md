# blurtrack_kd

TensorRT-friendly student distillation pipeline for 3-frame ping-pong blur tracking.

## Install

```bash
pip install -r requirements.txt
```

## 2.1 Build index

```bash
python -m src.datasets.frame_window_dataset --build_index \
  --frames_root /home/lht/blurtrack/video_maked --out data/index.jsonl \
  --mode predict_middle \
  --pseudo_root /home/lht/blurtrack/pseudo/heatmaps
```
This supports real layout: `frames_root/{segment_id}/frames_roi/00000.jpg`, and writes `img_paths` + direct `label_npz` path in each jsonl row.

## 2.2 Teacher pseudo generation (ONNX or FakeTeacher demo)

```bash
python -m src.teacher.teacher_runner \
  --index data/index.jsonl \
  --frames_root data/frames \
  --teacher_onnx path/to/teacher.onnx \
  --out_dir data/pseudo \
  --stride 4 \
  --gamma 2.0 \
  --save_backend npz
```

If `--teacher_onnx` is omitted, a built-in `FakeTeacher` is used for demo.

Outputs:
- `data/pseudo/heatmaps/*.npz` (or zarr)
- `data/pseudo/moments.csv`
- `data/pseudo/meta.json`

## 2.3 Train student

```bash
python -m src.train --config configs/student_rep_unet_s.yaml
```

## 2.4 Export ONNX params-only

```bash
python -m src.export_onnx \
  --ckpt outputs/best.pt \
  --out outputs/student_params.onnx \
  --static_shape 1 9 288 512 \
  --export_params_only 1
```

## 2.5 Visualize debug

```bash
python tools/visualize_debug.py \
  --frames_root data/frames --index data/index.jsonl \
  --pseudo_dir data/pseudo \
  --ckpt outputs/best.pt \
  --n 20 --out outputs/debug_vis
```

## 2.6 TensorRT optional

```bash
python tools/build_trt.py --onnx outputs/student_params.onnx --engine outputs/student_fp16.plan --fp16 1
python tools/benchmark_trt.py --engine outputs/student_fp16.plan --warmup 200 --iters 2000
```

If TensorRT is unavailable the scripts exit gracefully with a message.
