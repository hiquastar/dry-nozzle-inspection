"""
eval_angle_reeval.py - re-score existing checkpoints on the corrected angle
metric, next to the two trivial baselines, with bootstrap CIs.

Why this exists (2026-09-16):
  * compute_spray_angle used to measure the angle in the normalised plane,
    which squashes the 4:3 frame to 1:1.  Reported degrees were not physical
    degrees.  Every old MAE must be recomputed on the pixel-plane metric so
    old and new models are compared in the same unit.
  * A MAE number is meaningless without the two "no-learning" baselines on
    the same test set: predict the train mean (constant) and predict the
    train mean of the nozzle at that position (position lookup).  These used
    to be computed by hand; this script makes them reproducible.

Both units are reported side by side ("squashed" = old behaviour,
"pixel" = corrected) so the thesis can state the correction explicitly.

Only checkpoints whose architecture matches the canonical model.py are
scored.  abl_B and abl_D were trained with their own model.py (absent from
the tree) - loading them with the canonical decoder gives garbage
(N_matched = 0 in chapter6_final_numbers.json is that garbage).

Output: results/angle_reeval.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import RESULTS_DIR, load_model  # also puts ../structure on sys.path
from dataset import AFGNetDataset
from metrics import compute_spray_angle, decode_predictions, tip_match

from config import ANGLE_ASPECT, ANGLE_MATCH_OBJ_THRESH, DEVICE, REPO_ROOT, SPRAY_IDX, TEST_JSON, TRAIN_JSON, VAL_JSON

RUNS = {
    "full": ("v9_fixes_reverse_seed42", "train_v9_v1.pth", "Full model"),
    "abl_A": ("v9_rabl_A_no_hier_seed42", "train_v9_v1_rabl_A.pth", "Abl A: no hierarchical masking"),
    "abl_C": ("v9_rabl_C_no_flipswap_seed42", "train_v9_v1_rabl_C.pth", "Abl C: no flip identity swap"),
}
UNITS = {"squashed": 1.0, "pixel": ANGLE_ASPECT}
CLUSTER_RADIUS = 0.06  # normalised; nozzle clusters are ~0.005 wide, >0.10 apart
N_BOOT = 2000
OUT = RESULTS_DIR / "angle_reeval.json"


def spray_instances(ds):
    """(box_centre_xy, kpts) for every GT spray with valid keypoints."""
    out = []
    for s in ds.samples:
        for b, c, k in zip(s["gt_boxes"], s["gt_classes"], s["gt_kpts"]):
            if c == SPRAY_IDX and any(v > 0 for v in k):
                out.append((np.array(b[:2]), np.array(k)))
    return out


def cluster_centres(points, radius):
    """Greedy clustering: a point joins the first centre within `radius`."""
    centres = []
    for p in points:
        for c in centres:
            if np.linalg.norm(p - c["sum"] / c["n"]) <= radius:
                c["sum"] += p
                c["n"] += 1
                break
        else:
            centres.append({"sum": p.copy(), "n": 1})
    return [c["sum"] / c["n"] for c in centres]


def nearest_cluster(p, centres, radius):
    d = [np.linalg.norm(p - c) for c in centres]
    i = int(np.argmin(d))
    return i if d[i] <= radius else None


def bootstrap_ci(errors, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    e = np.asarray(errors)
    if e.size == 0:
        return [float("nan"), float("nan")]
    means = [rng.choice(e, size=e.size, replace=True).mean() for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarise(errors, extra=None):
    e = np.asarray(errors, dtype=float)
    d = {
        "MAE_deg": float(e.mean()) if e.size else float("nan"),
        "CI95": bootstrap_ci(e),
        "N": int(e.size),
    }
    if extra:
        d.update(extra)
    return d


def baselines(train_ds, test_ds, aspect):
    tr = spray_instances(train_ds)
    te = spray_instances(test_ds)
    tr_ang = np.array([compute_spray_angle(k, aspect) for _, k in tr])
    te_ang = np.array([compute_spray_angle(k, aspect) for _, k in te])

    const = float(tr_ang.mean())
    centres = cluster_centres([p for p, _ in tr], CLUSTER_RADIUS)
    cluster_mean = []
    for c in centres:
        idx = [i for i, (p, _) in enumerate(tr) if np.linalg.norm(p - c) <= CLUSTER_RADIUS]
        cluster_mean.append(float(tr_ang[idx].mean()))

    const_err, lookup_err, n_unmatched = [], [], 0
    for (p, _), a in zip(te, te_ang):
        const_err.append(abs(a - const))
        ci = nearest_cluster(p, centres, CLUSTER_RADIUS)
        if ci is None:
            n_unmatched += 1
            lookup_err.append(abs(a - const))
        else:
            lookup_err.append(abs(a - cluster_mean[ci]))

    return {
        "constant": summarise(const_err, {"predicts_deg": const}),
        "position_lookup": summarise(
            lookup_err,
            {
                "n_clusters": len(centres),
                "cluster_centres": [c.round(4).tolist() for c in centres],
                "cluster_mean_deg": [round(m, 2) for m in cluster_mean],
                "test_unmatched": n_unmatched,
            },
        ),
        "gt_test_mean_deg": float(te_ang.mean()),
    }


def model_errors(model, test_ds):
    """Per tip-matched GT spray: (gt_kpts, pred_kpts). Standard rule
    (metrics.tip_match): nearest spray prediction by tip distance within
    TIP_MATCH_PX, decoded at ANGLE_MATCH_OBJ_THRESH. Angle is computed later
    so one forward pass serves both units."""
    rows = []
    n_gt = 0
    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="  infer", leave=False, unit="img"):
            s = test_ds.samples[idx]
            gts = [i for i, c in enumerate(s["gt_classes"]) if c == SPRAY_IDX and any(v > 0 for v in s["gt_kpts"][i])]
            n_gt += len(gts)
            if not gts:
                continue
            img, _ = test_ds[idx]
            out = model(img.unsqueeze(0).to(DEVICE))
            preds = [p for p in decode_predictions(out, batch_idx=0, obj_thresh=ANGLE_MATCH_OBJ_THRESH) if p["class_id"] == SPRAY_IDX]
            for gi in gts:
                m = tip_match(preds, s["gt_kpts"][gi])
                if m is not None:
                    rows.append((np.array(s["gt_kpts"][gi]), np.array(m["keypoints"])))
    return rows, n_gt


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test",
                    help="val for tuning/comparison; test only once for the final model")
    ap.add_argument("--ckpt", default=None, help="score this checkpoint instead of the model-1 RUNS table")
    ap.add_argument("--label", default=None, help="name for --ckpt (default: parent dir name)")
    args = ap.parse_args()

    global OUT
    eval_json = VAL_JSON if args.split == "val" else TEST_JSON
    runs = RUNS
    tag = "v9_fixes_reverse_seed42"  # same tag eval_counterfactual derives for model 1
    if args.ckpt:
        p = Path(args.ckpt)
        tag = args.label or p.parent.name
        runs = {tag: (p, None, tag)}
    OUT = RESULTS_DIR / f"angle_reeval_{args.split}_{tag}.json"

    print(f"Loading splits...  (eval on {args.split})")
    train_ds = AFGNetDataset(TRAIN_JSON, augment=False)
    test_ds = AFGNetDataset(eval_json, augment=False)

    result = {"_units": {"squashed": "x/W, y/H plane (pre-2026-09-16)", "pixel": f"x scaled by W/H = {ANGLE_ASPECT:.4f}"}, "split": args.split}

    for unit, aspect in UNITS.items():
        result[unit] = {"baselines": baselines(train_ds, test_ds, aspect)}

    for key, (run_dir, ckpt_name, label) in runs.items():
        ckpt = Path(run_dir) if ckpt_name is None else REPO_ROOT / "train" / "model_1" / "checkpoint" / run_dir / ckpt_name
        if not ckpt.exists():
            print(f"SKIP {key}: {ckpt} missing")
            continue
        print(f"\n{label}")
        model = load_model(ckpt, DEVICE)
        rows, n_gt = model_errors(model, test_ds)
        for unit, aspect in UNITS.items():
            errs = []
            for gk, pk in rows:
                ga, pa = compute_spray_angle(gk, aspect), compute_spray_angle(pk, aspect)
                if ga is not None and pa is not None:
                    errs.append(abs(ga - pa))
            result[unit][key] = summarise(errs, {"label": label, "N_gt_spray": n_gt, "detection_rate": len(rows) / n_gt})

    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f'{"":<34}{"squashed (old)":>20}{"pixel (corrected)":>24}')
    print("-" * 78)
    rows_out = [("constant baseline", "baselines", "constant"), ("position-lookup baseline", "baselines", "position_lookup")]
    rows_out += [(runs[k][2], k, None) for k in runs if k in result["pixel"]]
    for name, k1, k2 in rows_out:
        cells = []
        for unit in UNITS:
            d = result[unit][k1][k2] if k2 else result[unit][k1]
            lo, hi = d["CI95"]
            cells.append(f'{d["MAE_deg"]:6.2f}° [{lo:5.2f},{hi:5.2f}]')
        print(f"{name:<34}{cells[0]:>20}{cells[1]:>24}")
    n = result["pixel"]["baselines"]["position_lookup"]
    print("-" * 78)
    print(f'lookup: {n["n_clusters"]} clusters, means {n["cluster_mean_deg"]}, test unmatched {n["test_unmatched"]}')
    full = result["pixel"].get("full")
    if full:
        print(f'model N matched = {full["N"]} / {full["N_gt_spray"]} GT sprays')
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
