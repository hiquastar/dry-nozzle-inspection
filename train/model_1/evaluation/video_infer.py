"""
video_infer.py - Run AFGNet v9 model on video file (real-time + save).

Run inference on each video frame, drawing boxes/keypoints/scores like infer_visualize.py.

Usage:
    # Basic: run + save + display
    python video_infer.py --video path/to/video.mp4

    # Customize threshold
    python video_infer.py --video video.mp4 --threshold 0.10

    # Save only, no display (run on server / batch)
    python video_infer.py --video video.mp4 --no-show

    # Skip frames to run faster
    python video_infer.py --video video.mp4 --skip 2

    # Process only the first N frames (quick test)
    python video_infer.py --video video.mp4 --max-frames 100

Controls (when --show):
    Q       quit
    SPACE   pause / resume
    S       save screenshot of current frame
    .       (dot) step 1 frame while paused
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "structure"))

from metrics import compute_spray_angle, decode_predictions
from model import AFGNet

from config import (
    DEVICE,
    IMG_SIZE,
    NMS_IOU_THRESH,
    OBJ_THRESH,
    REPO_ROOT,
)

# Class colors (RGB tuple) - same as infer_visualize.py
CLASS_COLORS = {
    0: (0, 200, 0),  # nozzle_spray (green)
    1: (255, 165, 0),  # nozzle_no_spray (orange)
    2: (255, 50, 50),  # nozzle_facing (red)
    3: (50, 100, 255),  # big_nozzle (blue)
}
CLASS_LABELS = ["SPRAY", "NO_SPRAY", "FACING", "BIG"]


# ============================================================================
# Model loading
# ============================================================================
def find_latest_v1_run():
    """Find most recent v9_v1 run (skip multi-scale runs)."""
    runs_dir = REPO_ROOT / "train" / "model_1" / "checkpoint"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs/ at {runs_dir}")

    v9_runs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("v9_")]
    )
    for rd in reversed(v9_runs):
        pth = next(rd.glob("*.pth"), None)
        if pth is None:
            continue
        ckpt = torch.load(pth, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # Skip multi-scale (has smooth_p layers)
        if not any("smooth_p" in k for k in state.keys()):
            return rd
    raise FileNotFoundError("No single-scale v9 run found")


def load_model(run_dir):
    """Load best model weights from run directory."""
    run_dir = Path(run_dir)
    candidates = list(run_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"Model not found: {run_dir}/*.pth")
    ckpt_path = candidates[0]

    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = AFGNet(pretrained=False).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    if isinstance(ckpt, dict):
        if "epoch" in ckpt:
            print(f"  Epoch: {ckpt['epoch']}")
        if "composite" in ckpt:
            print(f"  Composite: {ckpt['composite']:.4f}")

    return model


# ============================================================================
# Frame processing
# ============================================================================
# Pre-build transform once (avoid recreating per frame)
_transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_frame(frame_bgr):
    """BGR frame -> normalized tensor [1, 3, H, W]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return _transform(rgb).unsqueeze(0)


@torch.no_grad()
def infer_frame(model, frame_bgr, threshold):
    """Run model on one frame, return list of detections."""
    tensor = preprocess_frame(frame_bgr).to(DEVICE)
    preds = model(tensor)
    detections = decode_predictions(
        preds,
        batch_idx=0,
        obj_thresh=threshold,
        nms_iou=NMS_IOU_THRESH,
    )
    return detections


def compute_label_position(box_xyxy, frame_wh, text_wh, margin=8):
    """Compute label position on the side of the box facing the image center.

    Logic:
      1. Compute box center (bcx, bcy) and image center (icx, icy)
      2. Direction box -> image center indicates which side label should appear
      3. Place label opposite to box edge nearest to image center

    Args:
        box_xyxy: (x1, y1, x2, y2) in pixels
        frame_wh: (frame_w, frame_h)
        text_wh: (text_w, text_h) of label text bbox
        margin: pixels gap between box and label

    Returns:
        (text_origin_x, text_origin_y): bottom-left corner for cv2.putText
        (bg_x1, bg_y1, bg_x2, bg_y2): rectangle for label background
    """
    x1, y1, x2, y2 = box_xyxy
    fw, fh = frame_wh
    tw, th = text_wh

    bcx = (x1 + x2) / 2
    bcy = (y1 + y2) / 2
    icx = fw / 2
    icy = fh / 2

    # Direction from box center to image center (sign matters)
    dx = icx - bcx  # positive = image center is to the RIGHT of box
    dy = icy - bcy  # positive = image center is BELOW box

    # Determine vertical placement (above or below box)
    # Place label on the side TOWARD image center, i.e. between box and center
    if dy > 0:
        # Image center is below box → label below box (between box and center)
        bg_y1 = y2 + margin
        bg_y2 = bg_y1 + th + 6
    else:
        # Image center is above box → label above box
        bg_y2 = y1 - margin
        bg_y1 = bg_y2 - th - 6

    # Determine horizontal placement
    # Label horizontal alignment based on box position relative to center
    if dx > 0:
        # Image center is to the right → align label LEFT edge to box LEFT
        bg_x1 = x1
    else:
        # Image center is to the left → align label RIGHT edge to box RIGHT
        bg_x1 = x2 - tw - 4

    bg_x2 = bg_x1 + tw + 4

    # Clamp to frame bounds (label always inside frame)
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

    # Text origin (bottom-left for cv2.putText)
    tx = int(bg_x1 + 2)
    ty = int(bg_y2 - 4)

    return (tx, ty), (int(bg_x1), int(bg_y1), int(bg_x2), int(bg_y2))


def draw_predictions_on_frame(frame_bgr, detections):
    """Draw boxes/keypoints/scores on frame. Labels point toward image center.
    Modifies frame in-place.
    """
    h, w = frame_bgr.shape[:2]

    # Sort by box area (large first) so small detections drawn LAST = on top
    # This helps small boxes' labels not be covered by big_nozzle's drawings.
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

        # Pixel coords
        x1 = int(x1n * w)
        y1 = int(y1n * h)
        x2 = int(x2n * w)
        y2 = int(y2n * h)

        color_rgb = CLASS_COLORS.get(cls_id, (255, 255, 255))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # cv2 BGR

        # Bounding box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_bgr, 2)

        # Spray keypoints + cone
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

        # Build label text
        label = CLASS_LABELS[cls_id]
        text = f"{label} {score:.2f}"
        if cls_id == 0:
            angle = compute_spray_angle(kpts)
            if angle is not None:
                text = f"{label} {score:.2f} | {angle:.1f} deg"
        if cls_id in (0, 2):
            text += f" | obs={obs:.2f}"

        # Compute label placement (toward image center)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (tx, ty), (bg_x1, bg_y1, bg_x2, bg_y2) = compute_label_position(
            (x1, y1, x2, y2),
            (w, h),
            (tw, th),
            margin=4,
        )

        # Optional: line connecting box to label (visual link)
        bcx = (x1 + x2) // 2
        bcy = (y1 + y2) // 2
        lcx = (bg_x1 + bg_x2) // 2
        lcy = (bg_y1 + bg_y2) // 2
        # only draw if label is offset (not overlapping box edge)
        if not (x1 <= lcx <= x2 and y1 <= lcy <= y2):
            cv2.line(frame_bgr, (bcx, bcy), (lcx, lcy), color_bgr, 1, cv2.LINE_AA)

        # Filled background rectangle
        cv2.rectangle(frame_bgr, (bg_x1, bg_y1), (bg_x2, bg_y2), color_bgr, -1)
        # Text on top
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
    """Draw info overlay (top-left corner)."""
    h, w = frame_bgr.shape[:2]

    # Black semi-transparent background
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (380, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

    # Stats text
    pct = 100.0 * frame_idx / max(total_frames, 1)
    lines = [
        f"Frame: {frame_idx}/{total_frames} ({pct:.1f}%)",
        f"FPS: {fps:.1f}  |  Threshold: {threshold:.2f}",
        f"Detections: SP:{det_counts[0]} NS:{det_counts[1]} FA:{det_counts[2]} BG:{det_counts[3]}",
        f"Total: {sum(det_counts)}",
    ]
    if paused:
        lines.append("[PAUSED]  SPACE=resume  .=step  Q=quit")

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


# ============================================================================
# Main video processing
# ============================================================================
def process_video(model, video_path, args):
    """Main loop: read frames, infer, draw, save/display."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Video properties
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {video_path}")
    print(f"  Source FPS: {src_fps:.1f}")
    print(f"  Resolution: {src_w}x{src_h}")
    print(f"  Total frames: {total_frames}")
    if args.skip > 1:
        print(f"  Skip every {args.skip} frames")
    if args.max_frames:
        print(f"  Max frames: {args.max_frames}")

    # Output writer
    writer = None
    out_path = None
    # Default: data/infer/ (REPO_ROOT/data/infer)
    default_out = REPO_ROOT / "data" / "infer"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_save:
        stem = video_path.stem
        out_path = out_dir / f"{stem}_v9_inference_thr{args.threshold:.2f}.mp4"
        # Use mp4v codec (most compatible)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            src_fps / max(args.skip, 1),
            (src_w, src_h),
        )
        print(f"  Output video: {out_path}")

    # CSV logger
    csv_path = None
    csv_file = None
    csv_writer = None
    if not args.no_csv:
        csv_path = out_dir / f"{video_path.stem}_v9_log_thr{args.threshold:.2f}.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "frame",
                "time_sec",
                "n_spray",
                "n_no_spray",
                "n_facing",
                "n_big",
                "mean_score",
                "max_score",
                "has_spray",
                "mean_spray_angle",
                "mean_obs",
            ]
        )
        print(f"  CSV log: {csv_path}")

    # Stats tracking (for the final plot)
    history = {
        "frame_idx": [],
        "n_spray": [],
        "n_no_spray": [],
        "n_facing": [],
        "n_big": [],
        "mean_score": [],
        "max_score": [],
        "mean_spray_angle": [],
    }

    # Display window setup
    window_name = "AFGNet v9 - Video Inference"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(src_w, 1280), min(src_h, 720))

    # Main loop
    frame_idx = 0
    processed = 0
    fps_meter = 0.0
    fps_smooth = 0.9  # EMA factor
    paused = False

    print("\nProcessing... (press Q to quit if --show)")
    t_start = time.time()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\nEnd of video.")
                    break

                # Skip frames (read but don't process)
                if args.skip > 1 and frame_idx % args.skip != 0:
                    frame_idx += 1
                    continue

                if args.max_frames and processed >= args.max_frames:
                    print(f"\nReached max_frames={args.max_frames}.")
                    break

                # Inference
                t_infer = time.time()
                detections = infer_frame(model, frame, args.threshold)
                infer_time = time.time() - t_infer

                # Update FPS (EMA)
                inst_fps = 1.0 / max(infer_time, 1e-6)
                fps_meter = (
                    fps_smooth * fps_meter + (1 - fps_smooth) * inst_fps
                    if fps_meter > 0
                    else inst_fps
                )

                # Count per class
                det_counts = [0, 0, 0, 0]
                scores = []
                spray_angles = []
                obs_vals = []
                for det in detections:
                    det_counts[det["class_id"]] += 1
                    scores.append(det["score"])
                    if det["class_id"] in (0, 2):
                        obs_vals.append(det["observability"])
                    if det["class_id"] == 0:
                        a = compute_spray_angle(det["keypoints"])
                        if a is not None:
                            spray_angles.append(a)

                # Draw on frame
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

                # Save frame to output video
                if writer is not None:
                    writer.write(frame)

                # CSV log
                if csv_writer is not None:
                    mean_sc = float(np.mean(scores)) if scores else 0.0
                    max_sc = float(np.max(scores)) if scores else 0.0
                    mean_a = float(np.mean(spray_angles)) if spray_angles else -1.0
                    mean_o = float(np.mean(obs_vals)) if obs_vals else -1.0
                    csv_writer.writerow(
                        [
                            frame_idx,
                            f"{frame_idx / src_fps:.3f}",
                            det_counts[0],
                            det_counts[1],
                            det_counts[2],
                            det_counts[3],
                            f"{mean_sc:.4f}",
                            f"{max_sc:.4f}",
                            1 if det_counts[0] > 0 else 0,
                            f"{mean_a:.2f}",
                            f"{mean_o:.4f}",
                        ]
                    )

                # Track history for plot
                history["frame_idx"].append(frame_idx)
                history["n_spray"].append(det_counts[0])
                history["n_no_spray"].append(det_counts[1])
                history["n_facing"].append(det_counts[2])
                history["n_big"].append(det_counts[3])
                history["mean_score"].append(float(np.mean(scores)) if scores else 0.0)
                history["max_score"].append(float(np.max(scores)) if scores else 0.0)
                history["mean_spray_angle"].append(
                    float(np.mean(spray_angles)) if spray_angles else float("nan"),
                )

                processed += 1
                frame_idx += 1

                # Print progress every 50 frames
                if processed % 50 == 0:
                    elapsed = time.time() - t_start
                    avg_fps = processed / elapsed
                    print(
                        f"  Frame {frame_idx}/{total_frames}  "
                        f"avg {avg_fps:.1f} FPS  "
                        f"inference {fps_meter:.1f} FPS  "
                        f"detections this frame: {sum(det_counts)}"
                    )
            else:
                # Paused - re-draw last frame with PAUSED indicator
                pass

            # Display
            if args.show:
                # Resize for display if too large
                disp = frame
                if src_w > 1280:
                    scale = 1280 / src_w
                    disp = cv2.resize(frame, None, fx=scale, fy=scale)
                cv2.imshow(window_name, disp)

                # Wait for key (1ms or 0ms if paused)
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    print("\nUser quit (Q).")
                    break
                elif key == ord(" "):
                    paused = not paused
                    state = "PAUSED" if paused else "RESUMED"
                    print(f"  [{state} at frame {frame_idx}]")
                elif key == ord("s") or key == ord("S"):
                    snap_path = (
                        out_dir / f"{video_path.stem}_screenshot_f{frame_idx}.png"
                    )
                    cv2.imwrite(str(snap_path), frame)
                    print(f"  Screenshot saved: {snap_path}")
                elif key == ord(".") and paused:
                    # Step 1 frame while paused: temporarily unpause for 1 iteration
                    # (Implementation: just unpause, will pause again next loop iter
                    #  if we don't override it. Simplest: just process one frame.)
                    # We hack this: process one frame then re-pause
                    paused = False
                    # Read + process one frame
                    ret, frame = cap.read()
                    if ret:
                        detections = infer_frame(model, frame, args.threshold)
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
                            paused=True,
                        )
                        if writer is not None:
                            writer.write(frame)
                        frame_idx += 1
                        processed += 1
                    paused = True

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\nSaved video: {out_path}")
        if csv_file is not None:
            csv_file.close()
            print(f"Saved CSV log: {csv_path}")
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_fps = processed / max(elapsed, 1e-6)
    print(f'\n{"=" * 60}')
    print(f"Processed {processed} frames in {elapsed:.1f}s")
    print(f"Average inference: {avg_fps:.1f} FPS")
    print(f'{"=" * 60}')

    return history, out_path, out_dir


# ============================================================================
# Plot timeline
# ============================================================================
def save_timeline_plot(history, video_path, threshold, out_dir):
    """Plot detection counts + scores over time. out_dir must be a valid Path."""
    if not history["frame_idx"]:
        print("No frames processed - skipping plot.")
        return

    matplotlib.use("Agg")
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    frames = history["frame_idx"]

    # Plot 1: Detection counts per class over time
    ax = axes[0]
    ax.plot(frames, history["n_spray"], label="spray", color="green", linewidth=1.5)
    ax.plot(
        frames, history["n_no_spray"], label="no_spray", color="orange", linewidth=1.5
    )
    ax.plot(frames, history["n_facing"], label="facing", color="red", linewidth=1.5)
    ax.plot(frames, history["n_big"], label="big_nozzle", color="blue", linewidth=1.5)
    ax.set_ylabel("Detections per frame", fontsize=11)
    ax.set_title(
        f"AFGNet v9 — Video Inference Timeline ({video_path.name}, threshold={threshold:.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Plot 2: Mean & max score over time
    ax = axes[1]
    ax.plot(
        frames, history["mean_score"], label="mean score", color="blue", linewidth=1
    )
    ax.plot(
        frames, history["max_score"], label="max score", color="darkred", linewidth=1
    )
    ax.axhline(
        y=threshold,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=f"threshold={threshold:.2f}",
    )
    ax.set_ylabel("Detection score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Plot 3: Spray angle over time (only for frames with spray)
    ax = axes[2]
    angles = np.array(history["mean_spray_angle"])
    valid_mask = ~np.isnan(angles)
    if valid_mask.any():
        valid_frames = np.array(frames)[valid_mask]
        valid_angles = angles[valid_mask]
        ax.scatter(valid_frames, valid_angles, s=8, alpha=0.6, color="green")
        ax.set_ylabel("Mean spray angle (deg)", fontsize=11)
    else:
        ax.text(
            0.5,
            0.5,
            "No spray detections in video",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
    ax.set_xlabel("Frame number", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    plot_path = Path(out_dir) / f"{video_path.stem}_v9_timeline_thr{threshold:.2f}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved timeline plot: {plot_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="AFGNet v9 video inference (real-time + save).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run directory with .pth (default: latest v9_v1)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Detection score threshold (default 0.10)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <repo>/data/infer)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Process every N-th frame (1=all, 2=every other)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N processed frames (test quick)",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Display real-time (default: ON)",
    )
    parser.add_argument(
        "--no-show", dest="show", action="store_false", help="Disable real-time display"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving annotated MP4"
    )
    parser.add_argument("--no-csv", action="store_true", help="Skip saving CSV log")
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip saving timeline plot"
    )
    args = parser.parse_args()

    # Validate video
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return 1

    print("=" * 70)
    print("AFGNet v9 — Video Inference")
    print("=" * 70)

    # Load model
    run_dir = Path(args.run) if args.run else find_latest_v1_run()
    print(f"Run dir: {run_dir}")
    model = load_model(run_dir)

    # Process video
    history, out_path, out_dir = process_video(model, video_path, args)

    # Save timeline plot
    if not args.no_plot:
        save_timeline_plot(history, video_path, args.threshold, out_dir)

    print(f"\nAll outputs in: {out_dir}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
