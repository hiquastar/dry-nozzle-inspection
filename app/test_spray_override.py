"""
test_spray_override.py - Quick check that the DL-spray-override logic works on a
real video, WITHOUT launching the full app / setting up manual zones.

It runs the real ONNX model, takes the real spray keypoints, and calls the SAME
helpers used in production (nozzle_inspection.modules.zone_metrics) to:
  1. compute the DL spray angle from keypoints, and
  2. decide whether DL would override CV for several CV angles.
It also saves a demo image with the 2 override spray lines drawn at each spray nozzle.

Usage (from repo root):
    python app/test_spray_override.py --video "data/videos/20250509_5fps.mp4"
    python app/test_spray_override.py --video clip.mp4 --frame 10 --conf 0.12
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nozzle_inspection.modules import zone_metrics as zm  # noqa: E402
from nozzle_inspection.modules.dl_inference import AFGNetInference  # noqa: E402

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__),
    "..",
    "train",
    "model_1",
    "checkpoint",
    "v9_fixes_reverse_seed42",
    "pi5_package",
    "afgnet_v9.onnx",
)


class _Cfg:
    """Minimal stand-in for ConfigLoader.get used by the override helper."""

    def __init__(self, d):
        self.d = d

    def get(self, key, default=None):
        cur = self.d
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur


class _Sys:
    def __init__(self, enabled=True, max_diff=10.0):
        self.config = _Cfg(
            {
                "dl": {
                    "spray_override_enabled": enabled,
                    "spray_override_max_cv_diff_deg": max_diff,
                }
            }
        )


def _draw_override(frame, spray_info, color=(0, 255, 0)):
    """Draw the 2 spray lines the overlay would draw (anchor -> each endpoint)."""
    anchor = spray_info["anchor_point"]
    for _p1, p2 in spray_info["lines"]:
        cv2.line(
            frame,
            (int(anchor[0]), int(anchor[1])),
            (int(p2[0]), int(p2[1])),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(frame, (int(p2[0]), int(p2[1])), 6, (255, 0, 255), -1)
    cv2.circle(frame, (int(anchor[0]), int(anchor[1])), 6, (0, 255, 255), -1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Test DL spray override on a video.")
    ap.add_argument("--video", required=True, help="Video path (use one WITH spray).")
    ap.add_argument("--model", default=_DEFAULT_MODEL, help="ONNX model path.")
    ap.add_argument("--frame", type=int, default=3, help="Which frame to inspect.")
    ap.add_argument("--conf", type=float, default=0.12, help="Objectness threshold.")
    ap.add_argument(
        "--save", default="spray_override_demo.png", help="Demo image path."
    )
    args = ap.parse_args(argv)

    dl = AFGNetInference(
        os.path.abspath(args.model), n_threads=4, obj_threshold=args.conf
    )
    if not dl.available:
        print(f"ERROR: ONNX model/runtime unavailable: {args.model}")
        return 1

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}")
        return 1
    frame = None
    for _ in range(max(1, args.frame)):
        ok, frame = cap.read()
        if not ok:
            break
    cap.release()
    if frame is None:
        print("ERROR: could not read the requested frame.")
        return 1

    detections = [d for d in dl.detect(frame) if d["class_id"] == 0]
    print(f"Frame {args.frame}: {len(detections)} spray detection(s) (class_id==0)\n")
    if not detections:
        print(
            "No spray detected here. Try a different --frame or a video with spray "
            "(e.g. data/videos/20250509_5fps.mp4)."
        )
        return 0

    sysobj = _Sys(enabled=True, max_diff=10.0)
    n_drawn = 0
    for i, det in enumerate(detections, 1):
        parsed = zm._dl_spray_from_keypoints(det["keypoints"])
        if parsed is None:
            print(f"[{i}] invalid keypoints, skipped")
            continue
        dl_angle, anchor, _lines = parsed
        gate = {"class_id": 0, "measure_spray": True, "keypoints": det["keypoints"]}
        print(
            f"[{i}] score={det['score']:.2f}  DL_angle={dl_angle:6.1f} deg  tip={anchor}  "
            f"(rule: |CV-DL|>10 deg -> DL, else CV)"
        )
        for label, cv_info in (
            ("CV missing", {"detected": False}),
            (
                f"CV={dl_angle - 5:.0f} (diff 5)",
                {"detected": True, "spray_angle_deg": dl_angle - 5},
            ),
            (
                f"CV={dl_angle - 25:.0f} (diff 25)",
                {"detected": True, "spray_angle_deg": dl_angle - 25},
            ),
            (
                f"CV={dl_angle + 25:.0f} (diff 25)",
                {"detected": True, "spray_angle_deg": dl_angle + 25},
            ),
        ):
            out = zm._maybe_apply_dl_spray_override(sysobj, dict(cv_info), gate)
            winner = "DL" if out.get("spray_source") == "dl_onnx" else "CV"
            print(
                f"      {label:>18} -> {winner} wins  shown={out.get('spray_angle_deg'):.1f} deg"
            )

        # Draw the override result (CV missing -> DL wins) onto the frame.
        out = zm._maybe_apply_dl_spray_override(sysobj, {"detected": False}, gate)
        if out.get("spray_source") == "dl_onnx":
            _draw_override(frame, out)
            cv2.putText(
                frame,
                f"DL SPRAY {out['spray_angle_deg']:.1f} deg",
                (int(anchor[0]) - 60, max(20, int(anchor[1]) - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            n_drawn += 1
        print()

    out_path = os.path.abspath(args.save)
    if cv2.imwrite(out_path, frame):
        print(f"Saved demo image ({n_drawn} spray cone(s) drawn): {out_path}")
    else:
        print(f"WARNING: failed to write {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
