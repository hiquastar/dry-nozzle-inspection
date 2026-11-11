#!/usr/bin/env python3
"""
benchmark.py - the deployment measurement of Chapter 5 (Section "Embedded
Deployment") for the final model, on the Raspberry Pi 5.

Protocol (Chapter 5): 20 warm-up frames, not counted; then 100 frames of an
operating video at 2048x1536, each timed with time.perf_counter through the
full pipeline

    decode (cv2.VideoCapture.read) -> preprocess -> forward (ONNX Runtime)
    -> decode + NMS -> readers (mist gate + angle reader)

Recorded (Table tab:ch5-deploy-metrics): time per stage (mean, median, 95th
percentile), sustained throughput, CPU load of the four cores (mean, peak),
resident memory of the process (mean, peak), SoC temperature (start, end,
peak), wall power (mean, read from the inline meter and passed with
--power-w). The CPU clock is also logged, to see thermal throttling.

After timing, the outputs are compared with reference/video_s{seed}.json, the
outputs of the same package on the development PC, so the Pi is shown to
report the same lances, states and angles.

    python3 benchmark.py --seed 42 --power-w 7.9 --power-idle-w 3.1

Output: results/pi5_s{seed}.json (summary + per-frame times)
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import FinalSystem, box_iou  # noqa: E402

STAGES = ("decode", "preprocess", "forward", "decode_nms", "readers", "total")


def read_file(p):
    try:
        return Path(p).read_text().strip().rstrip("\x00")
    except OSError:
        return None


def soc_temp():
    v = read_file("/sys/class/thermal/thermal_zone0/temp")
    return int(v) / 1000.0 if v else None


def cpu_mhz():
    v = read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return int(v) / 1000.0 if v else None


def throttled():
    try:
        return subprocess.check_output(["vcgencmd", "get_throttled"], timeout=2).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return None


def stats(a):
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max())}


def summary_of(dets):
    return [{"box": [round(v, 5) for v in d["box"]], "state": d["state"], "angle_deg": d.get("angle_deg")} for d in dets]


def compare(got, ref):
    """frames whose detections differ from the reference of the development PC"""
    n_det, state, box, ang = 0, 0, 0.0, 0.0
    for g, r in zip(got, ref):
        if len(g) != len(r):
            n_det += 1
            continue
        for b in r:  # each reference lance against the Pi detection that overlaps it most
            a = max(g, key=lambda d: box_iou(d["box"], b["box"]))
            box = max(box, float(np.abs(np.subtract(a["box"], b["box"])).max()))
            state += a["state"] != b["state"]
            if a["angle_deg"] is not None and b["angle_deg"] is not None:
                ang = max(ang, abs(a["angle_deg"] - b["angle_deg"]))
    return {"frames": len(ref), "frames_det_count_differs": n_det, "states_differ": state,
            "box_max_abs_diff": box, "angle_max_abs_diff_deg": ang}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42, choices=[42, 123])
    ap.add_argument("--video", default=str(HERE / "data" / "video_cam3_v2.mp4"))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4, help="ONNX Runtime intra-op threads")
    ap.add_argument("--power-w", type=float, default=None, help="mean wall power during the timed run (inline meter)")
    ap.add_argument("--power-idle-w", type=float, default=None, help="wall power of the idle board (inline meter)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    proc = psutil.Process()
    system = FinalSystem(HERE, seed=args.seed, threads=args.threads)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def one_frame():
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("video ended before warm-up + count frames")
        t1 = time.perf_counter()
        dets, t = system(frame)
        t2 = time.perf_counter()
        return dets, {"decode": t1 - t0, **t, "total": t2 - t0}

    env = {"temp_idle_c": soc_temp(), "throttled_before": throttled()}
    outputs = []
    for _ in range(args.warmup):
        dets, _ = one_frame()
        outputs.append(summary_of(dets))

    psutil.cpu_percent(percpu=True)  # starts the CPU-load interval
    proc.cpu_percent()
    times = {k: [] for k in STAGES}
    cpu_mean, cpu_core_max, proc_cpu, rss, temp, mhz = [], [], [], [], [], []
    temp_start = soc_temp()
    w0 = time.perf_counter()
    for _ in range(args.count):
        dets, t = one_frame()
        for k in STAGES:
            times[k].append(t[k] * 1000.0)
        outputs.append(summary_of(dets))
        per_core = psutil.cpu_percent(percpu=True)  # since the previous frame
        cpu_mean.append(float(np.mean(per_core)))
        cpu_core_max.append(float(np.max(per_core)))
        proc_cpu.append(proc.cpu_percent())
        rss.append(proc.memory_info().rss / 2**20)
        temp.append(soc_temp())
        mhz.append(cpu_mhz())
    wall = time.perf_counter() - w0
    temp_end = soc_temp()

    res = {
        "seed": args.seed,
        "protocol": {"warmup": args.warmup, "count": args.count, "video": Path(args.video).name,
                     "source_resolution": [src_w, src_h], "threads": args.threads},
        "time_ms": {k: stats(v) for k, v in times.items()},
        "fps": {"sustained": args.count / wall, "from_mean_total": 1000.0 / float(np.mean(times["total"]))},
        "cpu_percent_4_cores": {"mean": float(np.mean(cpu_mean)), "peak": float(np.max(cpu_mean)),
                                "peak_single_core": float(np.max(cpu_core_max))},
        "cpu_percent_process": {"mean": float(np.mean(proc_cpu)), "peak": float(np.max(proc_cpu))},
        "rss_mb": {"mean": float(np.mean(rss)), "peak": float(np.max(rss))},
        "soc_temp_c": {"idle": env["temp_idle_c"], "start": temp_start, "end": temp_end,
                       "peak": max((x for x in temp if x is not None), default=None)},
        "cpu_mhz": {"min": min((x for x in mhz if x is not None), default=None),
                    "max": max((x for x in mhz if x is not None), default=None)},
        "throttled": {"before": env["throttled_before"], "after": throttled()},
        "power_w": {"mean_run": args.power_w, "idle": args.power_idle_w},
        "detections_per_frame": float(np.mean([len(o) for o in outputs[args.warmup:]])),
        "platform": {"model": read_file("/proc/device-tree/model"), "machine": platform.machine(),
                     "python": platform.python_version(), "onnxruntime": ort.__version__,
                     "opencv": cv2.__version__, "numpy": np.__version__, "cpu_count": os.cpu_count(),
                     "ram_total_mb": psutil.virtual_memory().total / 2**20},
        "per_frame_ms": times,
    }
    ref_p = HERE / "reference" / f"video_s{args.seed}.json"
    if ref_p.exists():
        ref = json.load(open(ref_p))
        n = min(len(ref["outputs"]), len(outputs))
        res["agreement_with_pc"] = compare(outputs[:n], ref["outputs"][:n])

    out = Path(args.out) if args.out else HERE / "results" / f"pi5_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)

    print(f"\nAFGNet final model, seed {args.seed}, {args.count} frames after {args.warmup} warm-up, {args.threads} threads")
    print(f"{'stage':<12}{'mean':>9}{'median':>9}{'p95':>9}   ms")
    for k in STAGES:
        s = res["time_ms"][k]
        print(f"{k:<12}{s['mean']:9.1f}{s['median']:9.1f}{s['p95']:9.1f}")
    print(f"FPS sustained {res['fps']['sustained']:.2f}   CPU mean/peak {res['cpu_percent_4_cores']['mean']:.0f}/"
          f"{res['cpu_percent_4_cores']['peak']:.0f} %   RSS mean/peak {res['rss_mb']['mean']:.0f}/{res['rss_mb']['peak']:.0f} MB")
    t = res["soc_temp_c"]
    print(f"SoC temp start/end/peak {t['start']}/{t['end']}/{t['peak']} C   CPU MHz min/max "
          f"{res['cpu_mhz']['min']}/{res['cpu_mhz']['max']}   throttled {res['throttled']['after']}")
    if "agreement_with_pc" in res:
        print("agreement with the PC:", res["agreement_with_pc"])
    print("wrote", out)


if __name__ == "__main__":
    main()
