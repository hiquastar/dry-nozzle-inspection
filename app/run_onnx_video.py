"""
run_onnx_video.py - Run AFGNet v9 ONNX model on a video and draw results in the
EXACT same style as train/train_v9_v1/video_infer.py (the script that produced the
data/infer/*_v9_inference_thr*.mp4 clips), so the ONNX output can be compared
side-by-side with the PyTorch-rendered ones.

Only the inference backend differs (onnxruntime instead of torch). Preprocessing,
6-head decoding, NMS and ALL drawing code are kept identical to the torch version.

Usage (from repo root):
    python app/run_onnx_video.py --video "data/some_clip.mp4"
    python app/run_onnx_video.py --video clip.mp4 --threshold 0.10
    python app/run_onnx_video.py --video clip.mp4 --no-show          # batch
    python app/run_onnx_video.py --video clip.mp4 --model other.onnx

Output (default): data/infer/<stem>_v9_onnx_thr<thr>.mp4
Controls (--show): Q quit, SPACE pause/resume, S screenshot.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ── Constants (must match training-time config.py) ──────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = 640
NMS_IOU_THRESH = 0.50
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DEFAULT_MODEL = (
    REPO_ROOT / "train" / "model_1" / "checkpoint" / "v9_fixes_reverse_seed42" / "pi5_package" / "afgnet_v9.onnx"
)

# Class colors (RGB) - identical to video_infer.py
CLASS_COLORS = {
    0: (0, 200, 0),  # nozzle_spray (green)
    1: (255, 165, 0),  # nozzle_no_spray (orange)
    2: (255, 50, 50),  # nozzle_facing (red)
    3: (50, 100, 255),  # big_nozzle (blue)
}
CLASS_LABELS = ["SPRAY", "NO_SPRAY", "FACING", "BIG"]
_EXPECTED_OUTPUTS = [
    "centerness",
    "bbox",
    "class_logits",
    "keypoints",
    "observability",
    "iou_quality",
]


# ── ONNX inference (numpy only, no torch) ───────────────────────────────────
def build_session(onnx_path, n_threads: int = 4) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(max(1, n_threads))
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )


def preprocess_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame -> float32 tensor (1,3,640,640). Matches video_infer.py transform."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = rgb.astype(np.float32) * (1.0 / 255.0)
    img = (img - _MEAN) / _STD
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(img)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def box_iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-7)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float):
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([box_iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious < iou_thresh]
    return keep


def decode_predictions(preds: dict, obj_thresh: float, nms_iou: float = NMS_IOU_THRESH):
    """Decode 6-head ONNX output -> list of detections (normalized boxes/keypoints).

    Output format matches train/.../metrics.decode_predictions so the shared
    drawing code can be reused unchanged.
    """
    centerness = preds["centerness"][0, 0]
    bbox = preds["bbox"][0]
    iou_q = preds["iou_quality"][0, 0]
    cls_logits = preds["class_logits"][0]
    obs = preds["observability"][0, 0]
    keypoints = preds["keypoints"][0]

    obj_score_map = centerness * iou_q
    mask = obj_score_map >= obj_thresh
    if not mask.any():
        return []

    iy, ix = np.where(mask)
    candidates = []
    for r, c in zip(iy, ix):
        cx = float(bbox[0, r, c])
        cy = float(bbox[1, r, c])
        bw = float(bbox[2, r, c])
        bh = float(bbox[3, r, c])
        x1 = max(0.0, cx - bw / 2)
        y1 = max(0.0, cy - bh / 2)
        x2 = min(1.0, cx + bw / 2)
        y2 = min(1.0, cy + bh / 2)
        if x2 <= x1 or y2 <= y1:
            continue
        cls_probs = _sigmoid(cls_logits[:, r, c])
        cls_id = int(np.argmax(cls_probs))
        cls_conf = float(cls_probs[cls_id])
        obj_score = float(obj_score_map[r, c])
        candidates.append(
            {
                "box": (x1, y1, x2, y2),
                "score": obj_score * cls_conf,
                "obj_score": obj_score,
                "cls_conf": cls_conf,
                "class_id": cls_id,
                "class_probs": cls_probs.tolist(),
                "observability": float(obs[r, c]),
                "keypoints": keypoints[:, r, c].tolist(),
            }
        )

    if not candidates:
        return []
    boxes = np.array([c["box"] for c in candidates])
    scores = np.array([c["score"] for c in candidates])
    keep = nms(boxes, scores, nms_iou)
    return [candidates[i] for i in keep]


def compute_spray_angle(kpts):
    """Angle (deg) at tip between tip->left and tip->right keypoint vectors."""
    kpts = np.asarray(kpts, dtype=np.float64)
    if not np.all(np.isfinite(kpts)) or len(kpts) < 6:
        return None
    tip, left, right = kpts[0:2], kpts[2:4], kpts[4:6]
    v1, v2 = left - tip, right - tip
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = max(-1.0, min(1.0, float(np.dot(v1, v2) / (n1 * n2))))
    return float(np.degrees(np.arccos(cos_a)))


# ── Drawing (copied verbatim from train/train_v9_v1/video_infer.py) ─────────
def compute_label_position(box_xyxy, frame_wh, text_wh, margin=8):
    """Place label on the side of the box facing the image center."""
    x1, y1, x2, y2 = box_xyxy
    fw, fh = frame_wh
    tw, th = text_wh
    bcx = (x1 + x2) / 2
    bcy = (y1 + y2) / 2
    icx = fw / 2
    icy = fh / 2
    dx = icx - bcx
    dy = icy - bcy
    if dy > 0:
        bg_y1 = y2 + margin
        bg_y2 = bg_y1 + th + 6
    else:
        bg_y2 = y1 - margin
        bg_y1 = bg_y2 - th - 6
    if dx > 0:
        bg_x1 = x1
    else:
        bg_x1 = x2 - tw - 4
    bg_x2 = bg_x1 + tw + 4
    if bg_x1 < 0:
        bg_x1 = 0
        bg_x2 = bg_x1 + tw + 4
    if bg_x2 > fw:
        bg_x2 = fw
        bg_x1 = bg_x2 - tw - 4
    if bg_y1 < 0:
        bg_y1 = 0
        bg_y2 = bg_y1 + th + 6
    if bg_y2 > fh:
        bg_y2 = fh
        bg_y1 = bg_y2 - th - 6
    tx = int(bg_x1 + 2)
    ty = int(bg_y2 - 4)
    return (tx, ty), (int(bg_x1), int(bg_y1), int(bg_x2), int(bg_y2))


def draw_predictions_on_frame(frame_bgr, detections):
    """Draw boxes/keypoints/scores; labels point toward image center. In-place."""
    h, w = frame_bgr.shape[:2]

    def _area(d):
        x1, y1, x2, y2 = d["box"]
        return (x2 - x1) * (y2 - y1)

    detections_sorted = sorted(detections, key=_area, reverse=True)

    for det in detections_sorted:
        x1n, y1n, x2n, y2n = det["box"]
        cls_id = det["class_id"]
        score = det["score"]
        obs = det["observability"]
        kpts = det["keypoints"]

        x1 = int(x1n * w)
        y1 = int(y1n * h)
        x2 = int(x2n * w)
        y2 = int(y2n * h)

        color_rgb = CLASS_COLORS.get(cls_id, (255, 255, 255))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_bgr, 2)

        if cls_id == 0:
            tip_x = int(kpts[0] * w)
            tip_y = int(kpts[1] * h)
            left_x = int(kpts[2] * w)
            left_y = int(kpts[3] * h)
            right_x = int(kpts[4] * w)
            right_y = int(kpts[5] * h)
            cv2.line(frame_bgr, (tip_x, tip_y), (left_x, left_y), color_bgr, 1)
            cv2.line(frame_bgr, (tip_x, tip_y), (right_x, right_y), color_bgr, 1)
            cv2.circle(frame_bgr, (tip_x, tip_y), 4, (0, 255, 255), -1)
            cv2.circle(frame_bgr, (left_x, left_y), 3, (255, 0, 255), -1)
            cv2.circle(frame_bgr, (right_x, right_y), 3, (255, 0, 255), -1)

        label = CLASS_LABELS[cls_id]
        text = f"{label} {score:.2f}"
        if cls_id == 0:
            angle = compute_spray_angle(kpts)
            if angle is not None:
                text = f"{label} {score:.2f} | {angle:.1f} deg"
        if cls_id in (0, 2):
            text += f" | obs={obs:.2f}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (tx, ty), (bg_x1, bg_y1, bg_x2, bg_y2) = compute_label_position(
            (x1, y1, x2, y2),
            (w, h),
            (tw, th),
            margin=4,
        )

        bcx = (x1 + x2) // 2
        bcy = (y1 + y2) // 2
        lcx = (bg_x1 + bg_x2) // 2
        lcy = (bg_y1 + bg_y2) // 2
        if not (x1 <= lcx <= x2 and y1 <= lcy <= y2):
            cv2.line(frame_bgr, (bcx, bcy), (lcx, lcy), color_bgr, 1, cv2.LINE_AA)

        cv2.rectangle(frame_bgr, (bg_x1, bg_y1), (bg_x2, bg_y2), color_bgr, -1)
        cv2.putText(
            frame_bgr,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame_bgr


def draw_overlay_stats(
    frame_bgr, frame_idx, total_frames, fps, det_counts, threshold, paused=False
):
    """Top-left info overlay - identical layout to video_infer.py."""
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (380, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

    pct = 100.0 * frame_idx / max(total_frames, 1)
    lines = [
        f"Frame: {frame_idx}/{total_frames} ({pct:.1f}%)",
        f"FPS: {fps:.1f}  |  Threshold: {threshold:.2f}",
        f"Detections: SP:{det_counts[0]} NS:{det_counts[1]} FA:{det_counts[2]} BG:{det_counts[3]}",
        f"Total: {sum(det_counts)}",
    ]
    if paused:
        lines.append("[PAUSED]  SPACE=resume  Q=quit")
    y = 22
    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return frame_bgr


# ── Main video loop ─────────────────────────────────────────────────────────
def process_video(sess, input_name, output_names, video_path, args):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {video_path}")
    print(
        f"  Source FPS: {src_fps:.1f}   Resolution: {src_w}x{src_h}   Frames: {total_frames}"
    )

    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "data" / "infer")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    out_path = None
    if not args.no_save:
        out_path = out_dir / f"{video_path.stem}_v9_onnx_thr{args.threshold:.2f}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path), fourcc, src_fps / max(args.skip, 1), (src_w, src_h)
        )
        print(f"  Output video: {out_path}")

    window_name = "AFGNet v9 ONNX - Video Inference"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(src_w, 1280), min(src_h, 720))

    frame_idx = 0
    processed = 0
    fps_meter = 0.0
    fps_smooth = 0.9
    paused = False
    t_start = time.time()
    print("\nProcessing... (press Q to quit if --show)")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\nEnd of video.")
                    break
                if args.skip > 1 and frame_idx % args.skip != 0:
                    frame_idx += 1
                    continue
                if args.max_frames and processed >= args.max_frames:
                    print(f"\nReached max_frames={args.max_frames}.")
                    break

                t_infer = time.time()
                outs = sess.run(None, {input_name: preprocess_bgr(frame)})
                preds = dict(zip(output_names, outs))
                detections = decode_predictions(preds, args.threshold, NMS_IOU_THRESH)
                infer_time = time.time() - t_infer

                inst_fps = 1.0 / max(infer_time, 1e-6)
                fps_meter = (
                    fps_smooth * fps_meter + (1 - fps_smooth) * inst_fps
                    if fps_meter > 0
                    else inst_fps
                )

                det_counts = [0, 0, 0, 0]
                for det in detections:
                    det_counts[det["class_id"]] += 1

                draw_predictions_on_frame(frame, detections)
                draw_overlay_stats(
                    frame,
                    frame_idx,
                    total_frames,
                    fps_meter,
                    det_counts,
                    args.threshold,
                    paused=False,
                )

                if writer is not None:
                    writer.write(frame)

                processed += 1
                frame_idx += 1
                if processed % 50 == 0:
                    elapsed = time.time() - t_start
                    print(
                        f"  Frame {frame_idx}/{total_frames}  "
                        f"avg {processed / elapsed:.1f} FPS  "
                        f"inference {fps_meter:.1f} FPS  "
                        f"det this frame: {sum(det_counts)}"
                    )

            if args.show:
                disp = frame
                if src_w > 1280:
                    scale = 1280 / src_w
                    disp = cv2.resize(frame, None, fx=scale, fy=scale)
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    print("\nUser quit (Q).")
                    break
                elif key == ord(" "):
                    paused = not paused
                    print(
                        f'  [{"PAUSED" if paused else "RESUMED"} at frame {frame_idx}]'
                    )
                elif key in (ord("s"), ord("S")):
                    snap = (
                        out_dir / f"{video_path.stem}_onnx_screenshot_f{frame_idx}.png"
                    )
                    cv2.imwrite(str(snap), frame)
                    print(f"  Screenshot saved: {snap}")
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\nSaved video: {out_path}")
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f'\n{"=" * 60}')
    print(
        f"Processed {processed} frames in {elapsed:.1f}s  "
        f"(avg {processed / max(elapsed, 1e-6):.1f} FPS)"
    )
    print(f'{"=" * 60}')
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="AFGNet v9 ONNX video inference (same drawing as video_infer.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", type=str, required=True, help="Input video path.")
    parser.add_argument(
        "--model", type=str, default=str(_DEFAULT_MODEL), help="ONNX model path."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.10, help="Detection score threshold."
    )
    parser.add_argument("--threads", type=int, default=4, help="ONNX intra-op threads.")
    parser.add_argument(
        "--out-dir", type=str, default=None, help="Output dir (default data/infer)."
    )
    parser.add_argument("--skip", type=int, default=1, help="Process every N-th frame.")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Stop after N frames."
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Display (default ON).",
    )
    parser.add_argument(
        "--no-show", dest="show", action="store_false", help="Disable display."
    )
    parser.add_argument("--no-save", action="store_true", help="Skip saving MP4.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return 1
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: ONNX model not found: {model_path}")
        return 1

    print("=" * 70)
    print("AFGNet v9 - ONNX Video Inference")
    print("=" * 70)
    print(f"Model: {model_path}")
    sess = build_session(model_path, n_threads=args.threads)
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    if output_names != _EXPECTED_OUTPUTS:
        print(f"WARNING: ONNX outputs reordered: {output_names} (mapped by name)")

    process_video(sess, input_name, output_names, video_path, args)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
