"""
bench_latency.py — End-to-end pipeline latency breakdown.

Measures wall-clock time for EVERY stage of the video inference pipeline:
  decode -> resize+normalize -> forward -> head decode + NMS -> overlay draw

Output: table with p50/p95 per stage, sustained FPS, total over N frames.

Usage:
    # Default: 500 frames from CAM2_V2_5fps on whichever device is available
    python bench_latency.py

    # Force CPU (simulates Pi5 deploy target)
    python bench_latency.py --device cpu

    # Pick a different video / frame count
    python bench_latency.py --video data/videos/CAM2_V3_5fps.mp4 --n 300
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
    sys.path.insert(0, str(_PARENT.parent / "structure"))

from metrics import compute_spray_angle, decode_predictions
from model import AFGNet
from video_infer import draw_predictions_on_frame

from config import IMG_SIZE, NMS_IOU_THRESH, REPO_ROOT

CANONICAL_RUN = "v9_fixes_reverse_seed42"
CKPT_PATH = REPO_ROOT / "train" / "model_1" / "checkpoint" / CANONICAL_RUN / "train_v9_v1.pth"
DEFAULT_VIDEO = REPO_ROOT / "data" / "videos" / "CAM2_V2_5fps.mp4"


def percentile(values, p):
    return float(np.percentile(values, p)) if values else float("nan")


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    state.pop("head.cell_x", None)
    state.pop("head.cell_y", None)
    model = AFGNet(pretrained=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO))
    parser.add_argument(
        "--n", type=int, default=500, help="Number of frames to time (after warmup)"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument(
        "--device", type=str, default=None, choices=[None, "cpu", "cuda"]
    )
    parser.add_argument(
        "--out", type=str, default=str(_HERE / "bench_latency_results.json")
    )
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="Skip overlay drawing stage (model-only timing)",
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print(f"AFGNet v9 — End-to-end latency benchmark")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Video      : {args.video}")
    print(f"Frames     : {args.n} (after {args.warmup} warmup)")
    print(f"Threshold  : {args.threshold}")

    model = load_model(device)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.video}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    print(f"Source     : {src_w}x{src_h} @ {src_fps:.1f} FPS")
    print()

    # Stage timings (ms)
    t_decode = []
    t_prep = []
    t_forward = []
    t_post = []
    t_draw = []
    t_total = []
    n_dets_history = []

    use_cuda = device.type == "cuda"

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    print("Warming up...")
    for _ in range(args.warmup):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(tensor)
        sync()

    print("Timing...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    t_run_start = time.perf_counter()
    processed = 0
    while processed < args.n:
        # ---- STAGE 1: decode frame from video ------------------------------
        sync()
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            # Loop video if it's shorter than n
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        sync()
        t1 = time.perf_counter()

        # ---- STAGE 2: preprocess (BGR->RGB, ToTensor, Resize, Normalize) ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)
        sync()
        t2 = time.perf_counter()

        # ---- STAGE 3: model forward -----------------------------------------
        with torch.no_grad():
            preds = model(tensor)
        sync()
        t3 = time.perf_counter()

        # ---- STAGE 4: decode heads + NMS ------------------------------------
        detections = decode_predictions(
            preds,
            batch_idx=0,
            obj_thresh=args.threshold,
            nms_iou=NMS_IOU_THRESH,
        )
        sync()
        t4 = time.perf_counter()

        # ---- STAGE 5: draw overlay on original-resolution frame -------------
        if not args.no_draw:
            draw_predictions_on_frame(frame, detections)
        sync()
        t5 = time.perf_counter()

        t_decode.append((t1 - t0) * 1000)
        t_prep.append((t2 - t1) * 1000)
        t_forward.append((t3 - t2) * 1000)
        t_post.append((t4 - t3) * 1000)
        t_draw.append((t5 - t4) * 1000)
        t_total.append((t5 - t0) * 1000)
        n_dets_history.append(len(detections))

        processed += 1
        if processed % 50 == 0:
            avg_total = np.mean(t_total[-50:])
            print(f"  [{processed:4d}/{args.n}]  recent avg total {avg_total:6.2f} ms")

    cap.release()
    elapsed = time.perf_counter() - t_run_start
    sustained_fps = processed / max(elapsed, 1e-6)

    # ── Summary table ────────────────────────────────────────────────────────
    stages = [
        ("Frame decode      ", t_decode),
        ("Preprocess        ", t_prep),
        ("Model forward     ", t_forward),
        ("Decode + NMS      ", t_post),
        ("Draw overlay      ", t_draw),
    ]
    total_p50 = percentile(t_total, 50)

    print()
    print("=" * 70)
    print(f"LATENCY BREAKDOWN  —  {device}, {processed} frames, {src_w}x{src_h} input")
    print("=" * 70)
    print(f'{"Stage":<22}{"p50 ms":>10}{"p95 ms":>10}{"mean":>10}{"% total":>10}')
    print("-" * 70)
    for name, times in stages:
        p50 = percentile(times, 50)
        p95 = percentile(times, 95)
        mean = float(np.mean(times))
        pct = 100.0 * p50 / total_p50 if total_p50 > 0 else 0
        print(f"{name:<22}{p50:>10.2f}{p95:>10.2f}{mean:>10.2f}{pct:>9.1f}%")
    print("-" * 70)
    p50_tot = percentile(t_total, 50)
    p95_tot = percentile(t_total, 95)
    p99_tot = percentile(t_total, 99)
    mean_tot = float(np.mean(t_total))
    print(
        f'{"TOTAL":<22}{p50_tot:>10.2f}{p95_tot:>10.2f}{mean_tot:>10.2f}{"100.0":>9}%'
    )
    print()
    print(f"p99 total       : {p99_tot:.2f} ms")
    print(f"Sustained FPS   : {sustained_fps:.1f}")
    print(f"1/p50 (max FPS) : {1000.0/p50_tot:.1f}")
    print(f"Avg detections  : {np.mean(n_dets_history):.1f} per frame")

    # ── Targets check ────────────────────────────────────────────────────────
    print()
    print(
        f"Camera input rate target  (>=5 FPS) : "
        f'{"PASS" if sustained_fps >= 5 else "FAIL"}  ({sustained_fps:.1f} FPS)'
    )
    print(
        f"Latency target (<=500 ms p95)       : "
        f'{"PASS" if p95_tot <= 500 else "FAIL"}  ({p95_tot:.1f} ms)'
    )

    # ── Save JSON ────────────────────────────────────────────────────────────
    out = {
        "device": str(device),
        "video": args.video,
        "src_resolution": [src_w, src_h],
        "model_input_size": IMG_SIZE,
        "threshold": args.threshold,
        "n_frames": processed,
        "sustained_fps": sustained_fps,
        "stages_ms": {
            "decode": {
                "p50": percentile(t_decode, 50),
                "p95": percentile(t_decode, 95),
                "mean": float(np.mean(t_decode)),
            },
            "prep": {
                "p50": percentile(t_prep, 50),
                "p95": percentile(t_prep, 95),
                "mean": float(np.mean(t_prep)),
            },
            "forward": {
                "p50": percentile(t_forward, 50),
                "p95": percentile(t_forward, 95),
                "mean": float(np.mean(t_forward)),
            },
            "post": {
                "p50": percentile(t_post, 50),
                "p95": percentile(t_post, 95),
                "mean": float(np.mean(t_post)),
            },
            "draw": {
                "p50": percentile(t_draw, 50),
                "p95": percentile(t_draw, 95),
                "mean": float(np.mean(t_draw)),
            },
            "total": {"p50": p50_tot, "p95": p95_tot, "p99": p99_tot, "mean": mean_tot},
        },
        "avg_detections_per_frame": float(np.mean(n_dets_history)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
