import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import os
import sys
import shutil
import urllib.request
import cv2
import time
import torch
from ultralytics import YOLO

# ── FORCE CPU MODE (ROBUST DETECTION) ──────────────────────
# torch.cuda.is_available() returns True when NVIDIA drivers
# are installed, EVEN IF no usable GPU is present.
# torch.cuda.device_count() is more honest — it returns the
# actual number of usable devices. We require BOTH to agree,
# plus a sanity-check tensor op, before trusting CUDA.
def _detect_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        # device_count == 0 means "drivers say yes, hardware
        # says no" — the exact situation causing our error
        if torch.cuda.device_count() < 1:
            return "cpu"
        # Final sanity check: can we actually run on GPU?
        _ = torch.zeros(1).cuda()
        torch.cuda.synchronize()
        return "cuda"
    except Exception:
        return "cpu"

DEVICE = _detect_device()

# If we're falling back to CPU, hide CUDA entirely from
# every library that tries to autodetect it.
if DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Also override torch.cuda.is_available so Ultralytics'
    # internal checks don't override our choice
    torch.cuda.is_available = lambda: False

# ── PATCH torch.jit.load / torch.load FOR CPU-ONLY ─────────
# MobileCLIP's .ts file was saved on a CUDA machine, so its
# tensors have device='cuda:0' baked in. On a CPU-only PC,
# torch.jit.load fails with:
#   "Attempting to deserialize object on CUDA device 0 but
#    torch.cuda.device_count() is 0"
# Patch both load functions to force map_location='cpu'.
if DEVICE == "cpu":
    _orig_jit_load = torch.jit.load
    def _cpu_jit_load(f, map_location=None, *args, **kwargs):
        return _orig_jit_load(f, map_location="cpu",
                              *args, **kwargs)
    torch.jit.load = _cpu_jit_load

    _orig_torch_load = torch.load
    def _cpu_torch_load(f, map_location=None,
                        *args, **kwargs):
        return _orig_torch_load(f, map_location="cpu",
                                *args, **kwargs)
    torch.load = _cpu_torch_load

# ══════════════════════════════════════════════════════════
# APP STATE
# ══════════════════════════════════════════════════════════
state = {
    "model_path" : None,
    "model"      : None,
    "exported"   : None,
    "running"    : False,
    "cap"        : None,
}

FORMATS    = ["tflite", "onnx", "engine", "openvino", "coreml"]
PRECISIONS = ["FP32", "FP16", "INT8"]

FMT_INFO = {
    "tflite"  : "Android phone · Nothing 2a",
    "onnx"    : "Universal · Raspberry Pi · CPU",
    "engine"  : "NVIDIA Jetson · TensorRT GPU",
    "openvino": "Intel CPU · Neural Compute Stick",
    "coreml"  : "iPhone · iPad · Apple Silicon",
}

PREC_INFO = {
    "FP32": ("Full precision · largest · most accurate",
             "#f0f0f0", "#444"),
    "FP16": ("Half precision · 2x smaller · faster",
             "#f0f0f0", "#444"),
    "INT8": ("4x smaller · fastest · needs calibration",
             "#FAEEDA", "#633806"),
}

# ══════════════════════════════════════════════════════════
# MOBILECLIP CACHE — fixes the curl 23 / permission denied
# error by pre-downloading to a writable folder
# ══════════════════════════════════════════════════════════
MOBILECLIP_URL = (
    "https://github.com/ultralytics/assets/releases/"
    "download/v8.4.0/mobileclip2_b.ts"
)
MOBILECLIP_NAME = "mobileclip2_b.ts"

# Use the user's home folder — guaranteed writable, never
# blocked by OneDrive / Program Files permissions
CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".yolo_edge_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHED_MOBILECLIP = os.path.join(CACHE_DIR, MOBILECLIP_NAME)

def ensure_mobileclip(log_fn=print):
    """
    Make sure mobileclip2_b.ts is available in the current
    working directory before YOLOE.set_classes() needs it.

    Why this exists:
    - Ultralytics tries to download this file (~242 MB) with
      curl into the CWD on first set_classes() call.
    - If CWD is read-only / inside OneDrive / blocked by AV,
      curl fails with error 23 ("failed to write file").
    - We pre-download it ourselves to ~/.yolo_edge_cache/
      (always writable) then symlink/copy to CWD.
    """
    target = os.path.join(os.getcwd(), MOBILECLIP_NAME)

    # Already present in CWD? Done.
    if os.path.exists(target) and os.path.getsize(target) > 100_000:
        log_fn(f"✅ MobileCLIP found in CWD", "green")
        return True

    # Not in cache? Download.
    if not (os.path.exists(CACHED_MOBILECLIP)
            and os.path.getsize(CACHED_MOBILECLIP) > 100_000):
        log_fn("▶ First run — downloading MobileCLIP "
               "text encoder (~242 MB)...", "blue")
        log_fn(f"   Saving to: {CACHE_DIR}", "#888")
        log_fn("   This happens ONCE. Please wait...", "#888")
        try:
            # urllib respects Python's network stack and
            # writes wherever we tell it — no curl headaches
            urllib.request.urlretrieve(
                MOBILECLIP_URL, CACHED_MOBILECLIP)
            size_mb = (os.path.getsize(CACHED_MOBILECLIP)
                       / (1024 * 1024))
            log_fn(f"✅ Downloaded MobileCLIP "
                   f"({size_mb:.1f} MB)", "green")
        except Exception as e:
            log_fn(f"❌ MobileCLIP download failed: "
                   f"{str(e)[:200]}", "red")
            log_fn("   Manual fix: download this URL "
                   "in your browser →", "orange")
            log_fn(f"   {MOBILECLIP_URL}", "orange")
            log_fn(f"   Then save it to: {CACHE_DIR}",
                   "orange")
            return False
    else:
        log_fn("✅ MobileCLIP already cached", "green")

    # Copy from cache to CWD so Ultralytics finds it
    try:
        shutil.copy2(CACHED_MOBILECLIP, target)
        log_fn(f"✅ MobileCLIP ready in CWD", "green")
        return True
    except Exception as e:
        log_fn(f"⚠️  Couldn't copy to CWD: "
               f"{str(e)[:150]}", "orange")
        log_fn("   Trying to run from cache dir...",
               "orange")
        # Last resort: change CWD to cache dir
        try:
            os.chdir(CACHE_DIR)
            log_fn(f"✅ Switched CWD → {CACHE_DIR}",
                   "green")
            return True
        except Exception as e2:
            log_fn(f"❌ Final fallback failed: {e2}", "red")
            return False

# ══════════════════════════════════════════════════════════
# ROOT WINDOW
# ══════════════════════════════════════════════════════════
root = tk.Tk()
root.title("YOLO Edge Deployment Tool")
root.geometry("700x860")
root.configure(bg="#f5f5f5")
root.resizable(True, True)

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton",
    font=("Segoe UI", 10), padding=(10, 6))

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def make_card(parent, title, step_num):
    outer = tk.Frame(parent, bg="#f5f5f5")
    outer.pack(fill="x", padx=16, pady=6)
    card = tk.Frame(outer, bg="white",
        relief="solid", bd=1)
    card.pack(fill="x")
    hdr = tk.Frame(card, bg="white")
    hdr.pack(fill="x", padx=14, pady=(10, 6))
    tk.Label(hdr,
        text=f" {step_num} ",
        bg="#E6F1FB", fg="#0C447C",
        font=("Segoe UI", 9, "bold")
    ).pack(side="left")
    tk.Label(hdr,
        text=f"  {title}",
        bg="white", fg="#111",
        font=("Segoe UI", 10, "bold")
    ).pack(side="left")
    body = tk.Frame(card, bg="white")
    body.pack(fill="x", padx=14, pady=(0, 12))
    return body

def stat_card(parent, label, var):
    f = tk.Frame(parent, bg="#f5f5f5",
        relief="flat", bd=0)
    f.pack(side="left", padx=(0, 8),
           ipadx=14, ipady=8)
    tk.Label(f, textvariable=var,
        bg="#f5f5f5", fg="#111",
        font=("Segoe UI", 16, "bold")).pack()
    tk.Label(f, text=label,
        bg="#f5f5f5", fg="#888",
        font=("Segoe UI", 9)).pack()
    return f

def log(msg, color="black"):
    log_box.config(state="normal")
    log_box.insert("end", msg + "\n", color)
    log_box.tag_config(color, foreground=color)
    log_box.see("end")
    log_box.config(state="disabled")

def get_size_mb(path):
    """Get size in MB — works for both files AND folders."""
    if os.path.isdir(path):
        total = 0
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
        return total / (1024 * 1024)
    else:
        return os.path.getsize(path) / (1024 * 1024)

# ══════════════════════════════════════════════════════════
# SCROLLABLE CANVAS
# ══════════════════════════════════════════════════════════
canvas_frame = tk.Canvas(root, bg="#f5f5f5",
    highlightthickness=0)
scrollbar = ttk.Scrollbar(root,
    orient="vertical", command=canvas_frame.yview)
canvas_frame.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")
canvas_frame.pack(side="left", fill="both", expand=True)

main = tk.Frame(canvas_frame, bg="#f5f5f5")
canvas_window = canvas_frame.create_window(
    (0, 0), window=main, anchor="nw")

def on_configure(e):
    canvas_frame.configure(
        scrollregion=canvas_frame.bbox("all"))
    canvas_frame.itemconfig(
        canvas_window,
        width=canvas_frame.winfo_width())

main.bind("<Configure>", on_configure)
canvas_frame.bind("<Configure>", on_configure)

def on_mousewheel(e):
    canvas_frame.yview_scroll(
        int(-1 * (e.delta / 120)), "units")
canvas_frame.bind_all("<MouseWheel>", on_mousewheel)

# ══════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════
title_frame = tk.Frame(main, bg="#f5f5f5")
title_frame.pack(fill="x", padx=16, pady=(16, 8))
tk.Label(title_frame,
    text="YOLO Edge Deployment Tool",
    bg="#f5f5f5", fg="#111",
    font=("Segoe UI", 16, "bold")
).pack(side="left")
tk.Label(title_frame,
    text="  export → convert → detect",
    bg="#f5f5f5", fg="#888",
    font=("Segoe UI", 10)
).pack(side="left")

# ══════════════════════════════════════════════════════════
# STEP 1 — Upload Model
# ══════════════════════════════════════════════════════════
s1 = make_card(main, "Upload .pt model", "1")

model_var = tk.StringVar(value="No model selected")
model_label = tk.Label(s1,
    textvariable=model_var,
    bg="#f9f9f9", fg="#555",
    font=("Segoe UI", 10),
    relief="solid", bd=1,
    anchor="w", padx=10)
model_label.pack(fill="x", pady=(0, 8))

def browse_model():
    path = filedialog.askopenfilename(
        title="Select YOLO .pt model",
        filetypes=[("PyTorch model", "*.pt"),
                   ("All files", "*.*")])
    if not path:
        return
    state["model_path"] = path
    state["model"]      = None
    name = os.path.basename(path)
    size = os.path.getsize(path) / (1024 * 1024)
    model_var.set(f"  {name}   ({size:.1f} MB)")
    model_label.config(fg="#0C447C")
    log(f"✅ Model selected: {name}  ({size:.1f} MB)",
        "green")

ttk.Button(s1,
    text="Browse .pt file",
    command=browse_model
).pack(anchor="w")

# ══════════════════════════════════════════════════════════
# STEP 2 — Export Format
# ══════════════════════════════════════════════════════════
s2 = make_card(main, "Choose export format", "2")

fmt_var  = tk.StringVar(value="tflite")
fmt_btns = {}

fmt_row = tk.Frame(s2, bg="white")
fmt_row.pack(fill="x", pady=(0, 8))

fmt_desc = tk.Label(s2,
    text=FMT_INFO["tflite"],
    bg="#EAF3DE", fg="#27500A",
    font=("Segoe UI", 9),
    padx=8, pady=4)
fmt_desc.pack(anchor="w")

def select_fmt(f):
    fmt_var.set(f)
    fmt_desc.config(text=FMT_INFO[f])
    for k, b in fmt_btns.items():
        on = k == f
        b.config(
            bg="#E6F1FB" if on else "#f0f0f0",
            fg="#0C447C" if on else "#444",
            relief="solid" if on else "flat")

for f in FORMATS:
    b = tk.Button(fmt_row,
        text=f.upper(),
        bg="#E6F1FB" if f == "tflite" else "#f0f0f0",
        fg="#0C447C" if f == "tflite" else "#444",
        font=("Segoe UI", 9, "bold"),
        relief="solid" if f == "tflite" else "flat",
        bd=1, padx=10, pady=5,
        cursor="hand2",
        command=lambda x=f: select_fmt(x))
    b.pack(side="left", padx=(0, 6))
    fmt_btns[f] = b

# ══════════════════════════════════════════════════════════
# STEP 3 — Precision
# ══════════════════════════════════════════════════════════
s3 = make_card(main, "Choose precision", "3")

prec_var  = tk.StringVar(value="INT8")
prec_btns = {}

prec_row = tk.Frame(s3, bg="white")
prec_row.pack(fill="x", pady=(0, 8))

prec_desc = tk.Label(s3,
    text=PREC_INFO["INT8"][0],
    bg=PREC_INFO["INT8"][1],
    fg=PREC_INFO["INT8"][2],
    font=("Segoe UI", 9),
    padx=8, pady=4)
prec_desc.pack(anchor="w")

def select_prec(p):
    prec_var.set(p)
    info = PREC_INFO[p]
    prec_desc.config(text=info[0], bg=info[1], fg=info[2])
    for k, b in prec_btns.items():
        on = k == p
        b.config(
            bg="#FAEEDA" if (on and p == "INT8")
               else "#E6F1FB" if on
               else "#f0f0f0",
            fg="#633806" if (on and p == "INT8")
               else "#0C447C" if on
               else "#444",
            relief="solid" if on else "flat")

for p in PRECISIONS:
    b = tk.Button(prec_row,
        text=p,
        bg="#FAEEDA" if p == "INT8" else "#f0f0f0",
        fg="#633806" if p == "INT8" else "#444",
        font=("Segoe UI", 9, "bold"),
        relief="solid" if p == "INT8" else "flat",
        bd=1, padx=14, pady=5,
        cursor="hand2",
        command=lambda x=p: select_prec(x))
    b.pack(side="left", padx=(0, 6))
    prec_btns[p] = b

# ══════════════════════════════════════════════════════════
# STEP 4 — Convert
# ══════════════════════════════════════════════════════════
s4 = make_card(main, "Convert model", "4")

convert_status = tk.StringVar(value="Ready to convert")
tk.Label(s4,
    textvariable=convert_status,
    bg="white", fg="#888",
    font=("Segoe UI", 9)
).pack(anchor="w", pady=(0, 8))

progress = ttk.Progressbar(s4,
    mode="indeterminate", length=400)
progress.pack(fill="x", pady=(0, 8))

stats_frame = tk.Frame(s4, bg="white")
stats_frame.pack(fill="x", pady=(0, 8))

out_size  = tk.StringVar(value="—")
reduction = tk.StringVar(value="—")
inf_shape = tk.StringVar(value="—")
stat_card(stats_frame, "output size",    out_size)
stat_card(stats_frame, "size reduction", reduction)
stat_card(stats_frame, "output shape",   inf_shape)

def do_convert():
    if not state["model_path"]:
        log("❌ Please select a .pt model first!", "red")
        return

    fmt  = fmt_var.get()
    prec = prec_var.get()

    def run():
        try:
            convert_status.set("Loading model...")
            progress.start(10)

            # ── FIX: TFLite version check ──────────────
            if fmt == "tflite":
                try:
                    import tensorflow as tf
                    ver = tf.__version__.split(".")
                    major, minor = int(ver[0]), int(ver[1])
                    if (major, minor) >= (2, 20):
                        log("⚠️  TensorFlow too new for TFLite!",
                            "orange")
                        log(f"   Your TF: {tf.__version__}",
                            "orange")
                        log("   Need TF 2.15 or lower.",
                            "orange")
                        log("   Run in CMD then restart app:",
                            "orange")
                        log("   pip install tensorflow==2.15.0"
                            " --force-reinstall", "orange")
                        convert_status.set(
                            "⚠️ TF version conflict — see log")
                        progress.stop()
                        return
                    else:
                        log(f"✅ TensorFlow {tf.__version__}"
                            " — compatible", "green")
                except ImportError:
                    log("❌ TensorFlow not installed!", "red")
                    log("   pip install tensorflow==2.15.0",
                        "red")
                    progress.stop()
                    return

            # ── Load model ─────────────────────────────
            log(f"\n▶ Loading "
                f"{os.path.basename(state['model_path'])}...",
                "blue")

            # Auto-detect task from filename
            fname = os.path.basename(
                state["model_path"]).lower()
            auto_task = ("segment"
                         if "seg" in fname else "detect")
            log(f"   Task: {auto_task} (auto-detected)",
                "#888")

            model = YOLO(state["model_path"],
                         task=auto_task)
            state["model"] = model
            orig_size = get_size_mb(state["model_path"])

            log(f"✅ Model loaded · {orig_size:.1f} MB",
                "green")
            convert_status.set(
                f"Exporting {fmt.upper()} {prec}...")
            log(f"▶ Exporting {fmt.upper()} · {prec}...",
                "blue")

            # ── FIX: ONNX + INT8 two-step ─────────────
            if fmt == "onnx" and prec == "INT8":
                log("   ONNX INT8 uses two steps:", "blue")
                log("   Step 1 — export FP32 ONNX...",
                    "blue")

                fp32_path = str(model.export(
                    format="onnx", imgsz=320))
                log(f"   ✅ FP32: "
                    f"{os.path.basename(fp32_path)}",
                    "green")

                log("   Step 2 — quantize to INT8...",
                    "blue")
                try:
                    from onnxruntime.quantization import (
                        quantize_dynamic, QuantType)
                except ImportError:
                    log("❌ onnxruntime not found!", "red")
                    log("   pip install onnxruntime", "red")
                    progress.stop()
                    return

                int8_path = fp32_path.replace(
                    ".onnx", "_int8.onnx")
                quantize_dynamic(
                    model_input  = fp32_path,
                    model_output = int8_path,
                    weight_type  = QuantType.QInt8)

                exported = int8_path
                log("   ✅ INT8 quantization done!", "green")

            # ── All other combos ───────────────────────
            else:
                kwargs = {"format": fmt, "imgsz": 320}
                if prec == "FP16":
                    kwargs["half"] = True
                elif prec == "INT8":
                    kwargs["int8"] = True
                    kwargs["data"] = "coco8.yaml"

                exported = str(model.export(**kwargs))

            state["exported"] = exported

            # ── FIX: size works for FILE and FOLDER ────
            # OpenVINO exports a folder, others export a file
            exp_size = get_size_mb(exported)
            red      = (1 - exp_size / orig_size) * 100

            display = os.path.basename(exported)

            out_size.set(f"{exp_size:.1f} MB")
            reduction.set(f"{red:.0f}%")
            inf_shape.set("[1,320,320,3]")

            log(f"✅ Done! → {display}", "green")
            log(f"   {orig_size:.1f} MB → {exp_size:.1f} MB"
                f"  ({red:.0f}% smaller)", "green")
            convert_status.set(f"✅ {display}")

        except Exception as e:
            log(f"❌ Export failed: {str(e)[:300]}", "red")
            convert_status.set("❌ Export failed — see log")
        finally:
            progress.stop()

    threading.Thread(target=run, daemon=True).start()

ttk.Button(s4,
    text="⚡ Convert now",
    command=do_convert
).pack(anchor="w")

# ══════════════════════════════════════════════════════════
# STEP 5 — Run Detection
# ══════════════════════════════════════════════════════════
s5 = make_card(main, "Run detection", "5")

detect_classes = tk.StringVar(
    value="person, face, bottle, chair, laptop, phone, cup")

tk.Label(s5,
    text="What to detect (comma separated):",
    bg="white", fg="#555",
    font=("Segoe UI", 9)
).pack(anchor="w", pady=(0, 4))

tk.Entry(s5,
    textvariable=detect_classes,
    font=("Segoe UI", 10),
    relief="solid", bd=1
).pack(fill="x", pady=(0, 10))

det_stats = tk.Frame(s5, bg="white")
det_stats.pack(fill="x", pady=(0, 10))
fps_var   = tk.StringVar(value="—")
obj_var   = tk.StringVar(value="—")
frame_var = tk.StringVar(value="—")
stat_card(det_stats, "avg FPS",  fps_var)
stat_card(det_stats, "objects",  obj_var)
stat_card(det_stats, "frames",   frame_var)

btn_frame = tk.Frame(s5, bg="white")
btn_frame.pack(fill="x")

def start_detection():
    """
    Kicks off detection on a background thread.
    All heavy work (model load, MobileCLIP setup,
    set_classes) happens in the thread so the UI
    stays responsive.
    """
    classes = [c.strip() for c in
               detect_classes.get().split(",")
               if c.strip()]

    if not classes:
        log("❌ Enter at least one class to detect!", "red")
        return

    if not state["model_path"]:
        log("❌ Load a model first!", "red")
        return

    start_btn.config(state="disabled")
    stop_btn.config(state="normal")

    def setup_and_run():
        try:
            # ── Step 1: Make sure MobileCLIP is ready ──
            log("▶ Checking MobileCLIP text encoder...",
                "blue")
            if not ensure_mobileclip(log):
                log("❌ MobileCLIP setup failed — "
                    "cannot continue", "red")
                start_btn.config(state="normal")
                stop_btn.config(state="disabled")
                return

            # ── Step 2: Load model if needed ──────────
            if not state["model"]:
                log("▶ Loading YOLO model...", "blue")
                # ── FIX: Auto-detect task from filename ──
                # YOLOE has separate detection vs segmentation
                # variants. Loading "-seg" model with task=
                # "detect" (or vice versa) gives:
                # "'NoneType' object has no attribute 'names'"
                fname = os.path.basename(
                    state["model_path"]).lower()
                if "-seg" in fname or "seg" in fname:
                    auto_task = "segment"
                else:
                    auto_task = "detect"
                log(f"   Detected task: {auto_task} "
                    f"(from filename)", "#888")
                log(f"   Device: {DEVICE}", "#888")

                state["model"] = YOLO(
                    state["model_path"], task=auto_task)
                # Force model onto chosen device
                try:
                    state["model"].to(DEVICE)
                except Exception:
                    pass
                log("✅ Model loaded", "green")

            # ── Step 3: Set open-vocab classes ────────
            log(f"▶ Setting classes: {classes}", "blue")
            log("   (first time takes 10-30s — "
                "generating text embeddings)", "#888")
            # Make sure model is on the right device BEFORE
            # set_classes — text encoder follows model.device
            try:
                state["model"].to(DEVICE)
            except Exception:
                pass
            state["model"].set_classes(classes)
            log("✅ Classes ready", "green")

            # ── Step 4: Open webcam ───────────────────
            # On Windows, the default backend sometimes fails
            # with "Invalid device id". Try DirectShow then
            # MSMF then default, in that order.
            log("▶ Opening webcam...", "blue")
            state["cap"] = None
            for backend, name in [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF,  "MSMF"),
                (cv2.CAP_ANY,   "default"),
            ]:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    state["cap"] = cap
                    log(f"✅ Webcam opened via {name}",
                        "green")
                    break
                cap.release()

            if state["cap"] is None or not state["cap"].isOpened():
                log("❌ Webcam not found on any backend!",
                    "red")
                log("   Check: is another app using the "
                    "camera? (Zoom, Teams, browser tab)",
                    "orange")
                start_btn.config(state="normal")
                stop_btn.config(state="disabled")
                return

            state["running"] = True
            log(f"▶ Detection started · "
                f"{len(classes)} classes", "blue")
            log("   Press 'Q' in the webcam window "
                "to stop", "#888")

            # ── Step 5: The detection loop ────────────
            detected = set()
            counts   = {}
            t0       = time.time()
            fc       = [0]

            while state["running"]:
                ret, frame = state["cap"].read()
                if not ret:
                    break
                fc[0] += 1

                results   = state["model"].predict(
                    source=frame, imgsz=320,
                    conf=0.3, verbose=False,
                    device=DEVICE)
                annotated = results[0].plot()
                boxes     = results[0].boxes
                summary   = {}

                for box in boxes:
                    idx  = int(box.cls)
                    name = (classes[idx]
                            if idx < len(classes)
                            else "object")
                    summary[name] = summary.get(name, 0) + 1
                    detected.add(name)
                    counts[name] = counts.get(name, 0) + 1

                fps = fc[0] / (time.time() - t0)
                fps_var.set(f"{fps:.1f}")
                obj_var.set(str(len(detected)))
                frame_var.set(str(fc[0]))

                cv2.putText(annotated,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

                y = 60
                for n, c in summary.items():
                    cv2.putText(annotated,
                        f"{n}: {c}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
                    y += 28

                cv2.putText(annotated,
                    "YOLOE-26 | Nothing 2a Project",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

                cv2.imshow("YOLOE-26 | Q to quit",
                           annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ── Cleanup ───────────────────────────────
            state["running"] = False
            if state["cap"]:
                state["cap"].release()
            cv2.destroyAllWindows()
            start_btn.config(state="normal")
            stop_btn.config(state="disabled")
            update_results(counts)
            elapsed = time.time() - t0
            avg     = fc[0] / elapsed if elapsed > 0 else 0
            log(f"✅ Session done · {fc[0]} frames"
                f" · {avg:.1f} FPS", "green")

        except Exception as e:
            import traceback
            log(f"❌ Detection error: {str(e)[:300]}",
                "red")
            log("   Full traceback:", "red")
            for line in traceback.format_exc().splitlines()[-6:]:
                log(f"   {line}", "#999")
            state["running"] = False
            if state["cap"]:
                state["cap"].release()
            cv2.destroyAllWindows()
            start_btn.config(state="normal")
            stop_btn.config(state="disabled")

    threading.Thread(target=setup_and_run,
                     daemon=True).start()

def stop_detection():
    state["running"] = False
    log("⏹ Stopping...", "blue")

start_btn = ttk.Button(btn_frame,
    text="▶ Start webcam",
    command=start_detection)
start_btn.pack(side="left", padx=(0, 8))

stop_btn = ttk.Button(btn_frame,
    text="⏹ Stop",
    command=stop_detection,
    state="disabled")
stop_btn.pack(side="left")

# ══════════════════════════════════════════════════════════
# STEP 6 — Results
# ══════════════════════════════════════════════════════════
s6 = make_card(main, "Session results", "6")

results_frame = tk.Frame(s6, bg="white")
results_frame.pack(fill="x")

tk.Label(results_frame,
    text="Object",
    bg="#f5f5f5", fg="#555",
    font=("Segoe UI", 9, "bold"),
    width=20, anchor="w", padx=8
).grid(row=0, column=0, sticky="w")

tk.Label(results_frame,
    text="Total detections",
    bg="#f5f5f5", fg="#555",
    font=("Segoe UI", 9, "bold"),
    width=20, anchor="e", padx=8
).grid(row=0, column=1, sticky="e")

result_rows = []

def update_results(counts):
    for w in result_rows:
        w.destroy()
    result_rows.clear()
    for i, (name, count) in enumerate(
            sorted(counts.items(),
                   key=lambda x: x[1],
                   reverse=True)):
        bg = "#f9f9f9" if i % 2 == 0 else "white"
        l1 = tk.Label(results_frame,
            text=name, bg=bg, fg="#111",
            font=("Segoe UI", 10),
            anchor="w", padx=8, pady=4)
        l1.grid(row=i+1, column=0, sticky="ew")
        l2 = tk.Label(results_frame,
            text=f"{count:,}", bg=bg, fg="#555",
            font=("Segoe UI", 10),
            anchor="e", padx=8, pady=4)
        l2.grid(row=i+1, column=1, sticky="ew")
        result_rows.extend([l1, l2])

# ══════════════════════════════════════════════════════════
# LOG BOX
# ══════════════════════════════════════════════════════════
log_card = make_card(main, "Log", "→")
log_box = scrolledtext.ScrolledText(
    log_card,
    height=8,
    font=("Consolas", 9),
    bg="#1e1e1e", fg="#d4d4d4",
    relief="flat",
    state="disabled")
log_box.pack(fill="x")

log("YOLO Edge Deployment Tool ready", "green")
log(f"Working directory: {os.getcwd()}", "#888")
log(f"MobileCLIP cache:  {CACHE_DIR}", "#888")
log(f"Compute device:    {DEVICE.upper()}"
    + ("  (CPU-only mode — patched torch.load)"
       if DEVICE == "cpu" else ""), "#888")
log("", "#888")
log("Step 1: Browse your .pt model file", "#888")
log("Step 2: Choose format and precision", "#888")
log("Step 3: Click Convert then Start webcam", "#888")
log("", "#888")
log("Status on your Windows machine:", "#888")
log("  ✅ ONNX     — FP32 / FP16 / INT8", "green")
log("  ✅ OPENVINO — FP32 / FP16 / INT8", "green")
log("  ⚠️  TFLITE   — pip install tensorflow==2.15.0",
    "orange")
log("  ❌ ENGINE   — needs NVIDIA GPU", "red")
log("  ❌ COREML   — needs macOS", "red")

# ══════════════════════════════════════════════════════════
root.mainloop()