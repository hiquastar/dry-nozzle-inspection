"""
check_onnx_parity.py - check that the ONNX package for the Raspberry Pi 5
(deployment/pi5_final, numpy + onnxruntime) reports what the PyTorch system of
Chapter 6 reports, on every frame of the TEST set, for both seeds.

  PyTorch  fig_ch6_results.system() on the cached 640x480 frame (as every
           Chapter 6 number)
  ONNX     pipeline.FinalSystem on the SOURCE frame, resized on the fly as on
           the Pi (cv2.INTER_AREA, the resize that built the cache)

Per frame: the resized source against the cached frame, the number of
detections, and per matched detection the box, the reported state and the
angle of the angle reader. The error counts of Table tab:ch6-test-errors are
recomputed from the ONNX detections and compared with
results/test_errors_final_heads_s{seed}.json.

The Chapter 6 numbers come from the GPU, where cuDNN runs the convolutions in
TF32 (torch.backends.cudnn.allow_tf32 is True by default); the mist logits then
differ from float32 by about 4e-2, and lances whose gate value lies within
about 5e-4 of the threshold can change state. "cpu" runs PyTorch in float32 on
the CPU, the arithmetic of ONNX Runtime and of the Pi.

    python deploy/check_onnx_parity.py            (from train/model_2; PyTorch on the GPU)
    python deploy/check_onnx_parity.py cpu        (PyTorch in float32 on the CPU)

Output: results/onnx_parity_test_final{,_cpu}.json
"""

import os

for k, v in {"AFGNET_INSTANCE_NORM": "0", "AFGNET_TIP_RELATIVE_EDGES": "1",
             "AFGNET_MIST_HEAD": "1", "AFGNET_LEARNED_HEADS": "1"}.items():
    os.environ[k] = v

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "structure"))
sys.path.insert(0, str(HERE.parent / "evaluation"))

import cv2
import numpy as np

import eval_mist_hybrid as MH
import fig_ch6_results as FC
from common import RESULTS_DIR, load_model
from dataset import AFGNetDataset, cache_path_for, load_frame
from metrics import cxcywh_to_xyxy

from config import DEVICE, REPO_ROOT, TEST_JSON

PKG = REPO_ROOT / "deployment" / "pi5_final"
sys.path.insert(0, str(PKG))
from pipeline import FinalSystem  # noqa: E402


def count(cnt, s, dets):
    """one frame of FC.error_counts (same matching), into cnt"""
    src = FC.SOURCE_LABEL[s["source_video"]]
    for b, c in zip(s["gt_boxes"], s["gt_classes"]):
        g = cxcywh_to_xyxy(b)
        m = max(dets, key=lambda d: MH.box_iou(d["box"], g), default=None)
        if m is None or MH.box_iou(m["box"], g) <= 0.5:
            cnt[f"{src} {FC.NAME[c]}"]["not detected"] += 1
        elif m["state"] != c:
            cnt[f"{src} {FC.NAME[c]}"][f"reported {FC.NAME[m['state']]}"] += 1
        else:
            cnt[f"{src} {FC.NAME[c]}"]["correct"] += 1
    for d in dets:
        if all(MH.box_iou(d["box"], cxcywh_to_xyxy(b)) <= 0.5 for b in s["gt_boxes"]):
            cnt[f"{src} {FC.NAME[d['state']]}"]["unmatched detection"] += 1


def totals(cnt):
    t = collections.Counter()
    for v in cnt.values():
        t.update(v)
    return dict(t)


def run(seed, ds, device=DEVICE):
    model = load_model(REPO_ROOT / "train" / "model_2" / "checkpoint" / f"abl_heads_s{seed}" / "afgnet_v10.pth", device)
    thr = json.load(open(RESULTS_DIR / f"mist_hybrid_test_final_heads_s{seed}.json"))["state"]["gate_threshold_from_train"]
    onx = FinalSystem(PKG, seed=seed)
    assert abs(onx.thr - thr) < 1e-12
    cnt_t, cnt_o = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    r = {"frames": 0, "resize_max_abs_px": 0, "resize_frames_differ": 0, "n_det_differ": [], "state_differ": [], "box_differ": [],
         "box_max_abs": 0.0, "angle_max_abs_deg": 0.0, "gate_margin_min": None}
    for i, s in enumerate(ds.samples):
        src = cv2.imread(s["image_path"], cv2.IMREAD_COLOR)
        small = cv2.resize(src, (onx.W, onx.H), interpolation=cv2.INTER_AREA)
        cache = cv2.imread(str(cache_path_for(s["image_path"])), cv2.IMREAD_COLOR)
        dpx = int(np.abs(small.astype(int) - cache.astype(int)).max())
        r["resize_max_abs_px"] = max(r["resize_max_abs_px"], dpx)
        r["resize_frames_differ"] += dpx > 0

        _, _, dt = FC.system(model, ds, load_frame(s["image_path"]), thr)
        do, _ = onx(src)
        count(cnt_t, s, dt)
        count(cnt_o, s, do)
        r["frames"] += 1
        name = Path(s["image_path"]).name
        if len(dt) != len(do):
            r["n_det_differ"].append({"frame": name, "torch": len(dt), "onnx": len(do)})
        for a in dt:
            b = max(do, key=lambda d: MH.box_iou(d["box"], a["box"]), default=None)
            if b is None or MH.box_iou(a["box"], b["box"]) < 0.9:
                continue  # counted in n_det_differ
            db = float(np.abs(np.subtract(a["box"], b["box"])).max())
            r["box_max_abs"] = max(r["box_max_abs"], db)
            if db > 1e-3:  # more than float noise
                r["box_differ"].append({"frame": name, "max_abs": db, "class_torch": FC.NAME[a["class_head"]],
                                        "class_onnx": FC.NAME[b["class_id"]], "iou": MH.box_iou(a["box"], b["box"])})
            if a["state"] != b["state"]:
                r["state_differ"].append({"frame": name, "torch": FC.NAME[a["state"]], "onnx": FC.NAME[b["state"]],
                                          "gate": b.get("gate")})
            if "angle_head" in a and "angle_deg" in b:
                r["angle_max_abs_deg"] = max(r["angle_max_abs_deg"], abs(a["angle_head"] - b["angle_deg"]))
            if "gate" in b:
                g = abs(b["gate"] - thr)
                r["gate_margin_min"] = g if r["gate_margin_min"] is None else min(r["gate_margin_min"], g)
        if i % 250 == 0:
            print(f"  s{seed} {i}/{len(ds.samples)}", flush=True)

    saved = json.load(open(RESULTS_DIR / f"test_errors_final_heads_s{seed}.json"))["total"]
    r["errors_total_saved"], r["errors_total_torch"], r["errors_total_onnx"] = saved, totals(cnt_t), totals(cnt_o)
    r["errors_equal_saved"] = r["errors_total_onnx"] == saved
    r["errors_equal_torch"] = r["errors_total_onnx"] == r["errors_total_torch"]
    r["errors_per_group_equal"] = {k: dict(v) for k, v in cnt_o.items()} == {k: dict(v) for k, v in cnt_t.items()}
    print(json.dumps({k: v for k, v in r.items() if k not in ("n_det_differ", "state_differ", "box_differ")}, indent=1))
    print("  n_det_differ", len(r["n_det_differ"]), " state_differ", r["state_differ"], " box_differ", r["box_differ"])
    return r


def main():
    cpu = "cpu" in sys.argv[1:]
    if cpu:  # the PyTorch path of FC.system (MH.predict_map, MH.head_angle) on the CPU
        MH.DEVICE = "cpu"
    ds = AFGNetDataset(TEST_JSON, augment=False)
    out = {str(sd): run(sd, ds, MH.DEVICE) for sd in (42, 123)}
    p = RESULTS_DIR / f"onnx_parity_test_final{'_cpu' if cpu else ''}.json"
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
