"""
bench_temporal.py — Temporal consistency on continuous video.

Runs inference frame-by-frame, builds nozzle tracks via greedy IoU
association across consecutive frames, and reports per-track stability:

  - class_stability_type        : % consecutive frame pairs with same type     (big/small)
  - class_stability_activity    : % consecutive frame pairs with same activity (spray/no_spray/facing)
  - angle_jitter_std            : std-dev of spray angle within a track       (degrees)
  - angle_jitter_max            : max within-track |angle - median|           (degrees)
  - detection_persistence       : track length / span (max - min frame + 1)
  - spurious_flicker_rate       : tracks shorter than MIN_TRACK_LEN per minute (false alarms)

Tracking rule:
  Two detections in consecutive frames belong to same track if
    IoU(box_t, box_t-1) >= IOU_MATCH  AND  class subset compatible (see CLASS_GROUP_TYPE).

Class grouping for "type" stability (big_nozzle vs the small group):
  TYPE: {0,1,2}=small (spray/no_spray/facing), {3}=big
  ACTIVITY: {0,2}=active (spray/facing), {1}=inactive (no_spray), {3}=N/A

Usage:
    python bench_temporal.py
    python bench_temporal.py --video data/videos/CAM2_V3_5fps.mp4 --max-frames 500
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

from metrics import box_iou, compute_spray_angle, decode_predictions
from model import AFGNet

from config import IMG_SIZE, NMS_IOU_THRESH, REPO_ROOT

CANONICAL_RUN = "v9_fixes_reverse_seed42"
CKPT_PATH = REPO_ROOT / "train" / "model_1" / "checkpoint" / CANONICAL_RUN / "train_v9_v1.pth"
DEFAULT_VIDEO = REPO_ROOT / "data" / "videos" / "CAM2_V2_5fps.mp4"

IOU_MATCH = 0.30
MIN_TRACK_LEN = 3  # tracks shorter than this are "spurious flicker"


# Group class_id -> coarse labels
def type_group(cls_id):
    return "big" if cls_id == 3 else "small"


def activity_group(cls_id):
    if cls_id in (0, 2):
        return "active"
    if cls_id == 1:
        return "inactive"
    return "na"


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
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--out", type=str, default=str(_HERE / "bench_temporal_results.json")
    )
    parser.add_argument(
        "--plot", type=str, default=str(_HERE / "bench_temporal_timeline.png")
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_model(device)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("=" * 70)
    print(f"AFGNet v9 — Temporal consistency benchmark")
    print("=" * 70)
    print(f"Device : {device}")
    print(f"Video  : {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source : {src_fps:.1f} FPS, {total_frames} frames")

    # ── Pass 1: run inference per frame, collect detections ──────────────────
    print("\nInference pass...")
    t0 = time.time()
    all_dets = []  # list of (frame_idx, [det dict...])
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(tensor)
        dets = decode_predictions(
            preds,
            batch_idx=0,
            obj_thresh=args.threshold,
            nms_iou=NMS_IOU_THRESH,
        )
        all_dets.append((frame_idx, dets))
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  [{frame_idx}/{total_frames}]")
    cap.release()
    print(
        f"  Done. {frame_idx} frames in {time.time()-t0:.1f}s "
        f"({frame_idx/(time.time()-t0):.1f} FPS effective)"
    )

    # ── Pass 2: greedy IoU tracking ──────────────────────────────────────────
    print("\nTracking...")
    # Each track is a dict: {'id': int, 'history': [(frame, det)...], 'last_box': ...}
    next_track_id = 0
    active_tracks = []  # tracks updated within last few frames
    tracks = {}  # id -> track

    for frame, dets in all_dets:
        # Match each new det to an active track
        matched_track_ids = set()
        assignments = []
        for di, d in enumerate(dets):
            best_tid, best_iou = None, IOU_MATCH
            for t in active_tracks:
                if t["id"] in matched_track_ids:
                    continue
                # Only match within same coarse type (big vs small)
                if type_group(t["last_class"]) != type_group(d["class_id"]):
                    continue
                iou = box_iou(t["last_box"], d["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = t["id"]
            assignments.append(best_tid)
            if best_tid is not None:
                matched_track_ids.add(best_tid)

        # Update assigned tracks; start new tracks for unmatched dets
        for di, d in enumerate(dets):
            tid = assignments[di]
            if tid is None:
                tid = next_track_id
                next_track_id += 1
                tracks[tid] = {"id": tid, "history": []}
            tracks[tid]["history"].append((frame, d))
            tracks[tid]["last_box"] = d["box"]
            tracks[tid]["last_class"] = d["class_id"]
            tracks[tid]["last_frame"] = frame

        # Prune tracks not updated in last N frames
        active_tracks = [
            tracks[t["id"]] for t in tracks.values() if frame - t["last_frame"] <= 2
        ]

    print(f"  Total tracks: {len(tracks)}")

    # ── Pass 3: compute per-track stability ──────────────────────────────────
    track_stats = []
    for tid, t in tracks.items():
        hist = t["history"]
        if len(hist) < 2:
            track_stats.append(
                {
                    "id": tid,
                    "length": len(hist),
                    "span": 1,
                    "type_stability": None,
                    "activity_stability": None,
                    "angle_std": None,
                    "angle_max_dev": None,
                    "n_angles": 0,
                }
            )
            continue
        # Pairwise consecutive stability
        type_same = 0
        act_same = 0
        for (f1, d1), (f2, d2) in zip(hist[:-1], hist[1:]):
            if type_group(d1["class_id"]) == type_group(d2["class_id"]):
                type_same += 1
            if activity_group(d1["class_id"]) == activity_group(d2["class_id"]):
                act_same += 1
        n_pairs = len(hist) - 1
        # Angles for spray detections
        angles = []
        for f, d in hist:
            if d["class_id"] == 0:  # spray
                a = compute_spray_angle(d["keypoints"])
                if a is not None:
                    angles.append(a)
        angle_std = float(np.std(angles)) if len(angles) >= 2 else None
        angle_max_dev = (
            float(np.max(np.abs(np.array(angles) - np.median(angles))))
            if len(angles) >= 2
            else None
        )
        track_stats.append(
            {
                "id": tid,
                "length": len(hist),
                "span": hist[-1][0] - hist[0][0] + 1,
                "type_stability": type_same / n_pairs,
                "activity_stability": act_same / n_pairs,
                "angle_std": angle_std,
                "angle_max_dev": angle_max_dev,
                "n_angles": len(angles),
                "first_frame": hist[0][0],
                "last_frame": hist[-1][0],
            }
        )

    # ── Aggregate over tracks with length >= MIN_TRACK_LEN ───────────────────
    long_tracks = [s for s in track_stats if s["length"] >= MIN_TRACK_LEN]
    short_tracks = [s for s in track_stats if s["length"] < MIN_TRACK_LEN]

    duration_min = frame_idx / src_fps / 60.0
    spurious_per_min = len(short_tracks) / max(duration_min, 1e-6)

    def agg(vals):
        vals = [v for v in vals if v is not None]
        return (
            (float(np.mean(vals)), float(np.median(vals)))
            if vals
            else (float("nan"), float("nan"))
        )

    type_mean, type_med = agg([s["type_stability"] for s in long_tracks])
    act_mean, act_med = agg([s["activity_stability"] for s in long_tracks])
    angle_std_mean, angle_std_med = agg([s["angle_std"] for s in long_tracks])
    angle_dev_mean, angle_dev_med = agg([s["angle_max_dev"] for s in long_tracks])

    lengths = [s["length"] for s in long_tracks]
    detection_persistence = (
        float(np.mean([s["length"] / s["span"] for s in long_tracks]))
        if long_tracks
        else float("nan")
    )

    # ── Print summary ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"TEMPORAL CONSISTENCY  —  {Path(args.video).name}")
    print("=" * 70)
    print(f"  Frames evaluated              : {frame_idx}")
    print(
        f"  Video duration                : {duration_min:.2f} min "
        f"@ {src_fps:.1f} FPS"
    )
    print(f"  Total tracks                  : {len(tracks)}")
    print(f"  Tracks with length >= {MIN_TRACK_LEN}        : {len(long_tracks)}")
    print(
        f"  Median track length (frames)  : "
        f"{int(np.median(lengths)) if lengths else 0}"
    )
    print(
        f"  Mean track length (frames)    : "
        f"{float(np.mean(lengths)) if lengths else 0:.1f}"
    )
    print()
    print("  Class stability (consecutive-frame agreement, long tracks):")
    print(
        f"    type      (big/small)       : {type_mean:.3f} mean / {type_med:.3f} median"
    )
    print(
        f"    activity  (spray/no/fa)     : {act_mean:.3f} mean / {act_med:.3f} median"
    )
    print()
    print("  Angle jitter (within track, deg):")
    print(
        f"    std-dev                     : {angle_std_mean:.2f} mean / {angle_std_med:.2f} median"
    )
    print(
        f"    max deviation from median   : {angle_dev_mean:.2f} mean / {angle_dev_med:.2f} median"
    )
    print()
    print(
        f"  Detection persistence         : {detection_persistence:.3f}  "
        f"(track length / span)"
    )
    print(
        f"  Spurious flicker rate         : {spurious_per_min:.2f} short "
        f"tracks/min ({len(short_tracks)} total)"
    )

    # ── Plot timeline of longest track ───────────────────────────────────────
    if long_tracks:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            longest = max(long_tracks, key=lambda s: s["length"])
            tid = longest["id"]
            hist = tracks[tid]["history"]
            frames_arr = [h[0] for h in hist]
            angles_arr = []
            for f, d in hist:
                if d["class_id"] == 0:
                    a = compute_spray_angle(d["keypoints"])
                    angles_arr.append(a if a is not None else np.nan)
                else:
                    angles_arr.append(np.nan)
            scores_arr = [d["score"] for f, d in hist]
            classes_arr = [d["class_id"] for f, d in hist]

            fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
            axes[0].plot(frames_arr, classes_arr, "o-", color="C0", markersize=4)
            axes[0].set_ylabel("class_id\n(0=spray 1=no 2=fa 3=big)")
            axes[0].set_title(
                f"Longest track #{tid} — {len(hist)} frames, " f'span {longest["span"]}'
            )
            axes[0].set_ylim(-0.5, 3.5)
            axes[0].grid(alpha=0.3)

            axes[1].plot(frames_arr, scores_arr, "o-", color="C1", markersize=4)
            axes[1].set_ylabel("det score")
            axes[1].set_ylim(0, 1.05)
            axes[1].grid(alpha=0.3)

            axes[2].plot(frames_arr, angles_arr, "o-", color="C2", markersize=4)
            axes[2].set_ylabel("spray angle (°)")
            axes[2].set_xlabel("frame")
            axes[2].grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.plot, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"\nSaved timeline plot: {args.plot}")
        except Exception as e:
            print(f"\nPlot skipped: {e}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    out = {
        "video": args.video,
        "frames_evaluated": frame_idx,
        "duration_min": duration_min,
        "source_fps": src_fps,
        "n_tracks_total": len(tracks),
        "n_tracks_long": len(long_tracks),
        "min_track_length": MIN_TRACK_LEN,
        "iou_match_threshold": IOU_MATCH,
        "median_track_length": int(np.median(lengths)) if lengths else 0,
        "mean_track_length": float(np.mean(lengths)) if lengths else 0,
        "class_stability": {
            "type_mean": type_mean,
            "type_median": type_med,
            "activity_mean": act_mean,
            "activity_median": act_med,
        },
        "angle_jitter_deg": {
            "std_mean": angle_std_mean,
            "std_median": angle_std_med,
            "max_dev_mean": angle_dev_mean,
            "max_dev_median": angle_dev_med,
        },
        "detection_persistence": detection_persistence,
        "spurious_flicker_rate_per_min": spurious_per_min,
        "n_short_tracks": len(short_tracks),
        "per_track_stats": track_stats,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
