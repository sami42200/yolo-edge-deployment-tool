# YOLO Edge Deployment Tool

A desktop GUI that takes any YOLO `.pt` model and gets it running on edge devices in three clicks: **export → convert → detect**.

Built during my internship to make edge deployment less painful. Supports YOLO11, YOLO26, and YOLOE-26 (open-vocabulary) models.

![GUI Screenshot](docs/screenshot.png)

---

## What it does

1. **Upload** any YOLO `.pt` model (detection or segmentation, auto-detected from filename)
2. **Choose** an export format — TFLite, ONNX, OpenVINO, CoreML, or TensorRT Engine
3. **Choose** precision — FP32, FP16, or INT8
4. **Convert** the model with one click — see size reduction and output shape live
5. **Run** webcam detection on the converted model to verify it works
6. **Track** per-class detection counts across the session

## Real numbers (YOLOE-26n-seg)

| Format | Size | Reduction | Target hardware |
|---|---|---|---|
| Original `.pt` | 12.0 MB | baseline | — |
| ONNX FP32 | 5.9 MB | -50% | Universal / Raspberry Pi / CPU |
| ONNX INT8 | 3.7 MB | -69% | Universal / Raspberry Pi / CPU |
| TFLite INT8 | 3.6 MB | -70% | Android / Edge TPU |

Live detection: **~27 FPS on CPU** at 320×320 input.

---

## Why this exists (the gotchas)

Most "just export your model" tutorials skip the four problems that actually break edge deployment. This tool handles them:

### 1. `torch.cuda.is_available()` lies on Windows

On machines with NVIDIA drivers installed but no usable GPU (laptops with hybrid graphics, dead GPUs, etc.), `torch.cuda.is_available()` returns `True` even though `torch.cuda.device_count()` is `0`. Loading a model then crashes with `"Attempting to deserialize object on CUDA device 0"`.

**Fix:** Check `is_available()` AND `device_count() >= 1` AND run a sanity tensor op before trusting CUDA. If anything fails, hide CUDA entirely via `CUDA_VISIBLE_DEVICES=""` and monkey-patch `torch.cuda.is_available` to return `False`.

### 2. Ultralytics' INT8 ONNX export doesn't actually quantize

Passing `int8=True` to `model.export(format="onnx")` produces a model with INT8 *metadata* but FP32 weights. The file is the same size as FP32.

**Fix:** Two-step export. First export FP32 ONNX, then call `onnxruntime.quantization.quantize_dynamic` separately. The tool does this automatically when you pick ONNX + INT8.

### 3. MobileCLIP download fails on OneDrive / Program Files

YOLOE uses MobileCLIP for open-vocabulary text encoding. Ultralytics downloads `mobileclip2_b.ts` (~242 MB) into the current working directory using `curl`. If the CWD is read-only, on OneDrive, or blocked by antivirus, curl fails with error 23 ("failed to write file").

**Fix:** Pre-download the file via `urllib` to `~/.yolo_edge_cache/` (always writable), then copy it into the CWD. Falls back to changing CWD to the cache dir if the copy fails.

### 4. OpenCV's webcam backend on Windows is unreliable

`cv2.VideoCapture(0)` with the default backend randomly returns `"Invalid device id"` on Windows.

**Fix:** Try DirectShow → MSMF → default in order. Whichever opens first wins.

---

## INT8 calibration

INT8 quantization needs **calibration data** — a small set of representative images used to measure activation ranges in each layer. Without good calibration, INT8 accuracy collapses.

This repo includes `examples/calibration_sample.npy` — 20 sample images (128×128×3, float32) you can use as a starting point. For production, replace these with images that match your actual deployment scenario.

```python
import numpy as np
data = np.load("examples/calibration_sample.npy")
# shape: (20, 128, 128, 3), dtype: float32
```

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/YOUR_USERNAME/yolo-edge-deployment-tool.git
cd yolo-edge-deployment-tool
pip install -r requirements.txt
python run_yoloe.py
```

### Optional dependencies by export format

| Format | Extra install |
|---|---|
| TFLite | `pip install tensorflow==2.15.0` (newer TF breaks the export) |
| ONNX INT8 | `pip install onnxruntime` |
| OpenVINO | `pip install openvino` |
| TensorRT Engine | Requires NVIDIA GPU + TensorRT |
| CoreML | macOS only |

---

## Project structure

```
yolo-edge-deployment-tool/
├── run_yoloe.py              Main GUI application
├── requirements.txt          Python dependencies
├── README.md                 You are here
├── LICENSE                   MIT
├── docs/
│   ├── screenshot.png        GUI screenshot
│   └── demo.gif              Live detection demo
└── examples/
    └── calibration_sample.npy  20 sample images for INT8 calibration
```

---

## Status matrix (Windows CPU-only)

| Format | FP32 | FP16 | INT8 |
|---|---|---|---|
| ONNX | ✅ | ✅ | ✅ |
| OpenVINO | ✅ | ✅ | ✅ |
| TFLite | ⚠️ TF 2.15 only | ⚠️ TF 2.15 only | ⚠️ TF 2.15 only |
| Engine | ❌ Needs NVIDIA GPU | ❌ | ❌ |
| CoreML | ❌ Needs macOS | ❌ | ❌ |

---

## License

MIT. See `LICENSE`.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO framework
- [Apple ML Research](https://github.com/apple/ml-mobileclip) for MobileCLIP
- ONNX Runtime team for the quantization toolkit
