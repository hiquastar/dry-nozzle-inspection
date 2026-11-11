"""
compare_1e.py - step 1e: put model 1 and the v10 reference side by side on
VAL, on the standard metrics, and apply the four pre-registered "better"
rules from docs/notes/PLAN-V10.md section 6:

  1. cls F1 (macro, instance level) goes up
  2. angle MAE (tip-matched, pixel deg) not worse by more than 1 deg
  3. shift test not worse (mean MAE increase over shift_x/y +-2/4/6)
  4. stretch slope higher than model 1 (-0.23 on val)

Reads:
  results/angle_reeval_val_<tag>.json       (eval_angle_reeval.py --split val)
  results/counterfactual_val_<tag>.json     (eval_counterfactual.py --split val)
  <checkpoint dir>/metrics.json             (training log, best epoch)
Writes results/compare_1e.json and prints the table.
"""

import json
import sys
from pathlib import Path

import numpy as np

from common import CHECKPOINT_DIR, RESULTS_DIR  # noqa: F401

MODEL1 = "v9_fixes_reverse_seed42"
# tags to compare against model 1; override on the command line:
#   python compare_1e.py v10_ref_s42 v10_ref_s123 abl_Foff_s42 abl_Foff_s123
V10 = sys.argv[1:] or ["v10_ref_s42", "v10_ref_s123"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def shift_increase(cf):
    v = [d["MAE_increase_vs_base"] for k, d in cf["shift_rot"].items() if k.startswith("shift")]
    return float(np.mean(v)) if v else float("nan")


def row(tag):
    ar = load(RESULTS_DIR / f"angle_reeval_val_{tag}.json")
    cf = load(RESULTS_DIR / f"counterfactual_val_{tag}.json")
    r = {"tag": tag}
    if ar:
        key = [k for k in ar["pixel"] if k != "baselines"]
        if key:
            d = ar["pixel"][key[0]]
            r["angle_mae"] = d["MAE_deg"]
            r["angle_ci"] = d["CI95"]
            r["angle_n"] = d["N"]
            r["tip_match_rate"] = d["detection_rate"]
        r["lookup_baseline"] = ar["pixel"]["baselines"]["position_lookup"]["MAE_deg"]
    if cf:
        r["stretch_slope"] = cf["stretch"]["slope"]
        r["stretch_ci"] = cf["stretch"]["slope_CI95"]
        r["slope_y"] = cf.get("stretch_y", {}).get("slope")
        r["slope_shear"] = cf.get("shear", {}).get("slope")
        r["spurious"] = cf["stretch"]["spurious_pred_change_deg"]
        r["shift_increase"] = shift_increase(cf)
        r["mask_cone_nopred"] = f'{cf["mask"]["cone"]["no_spray_prediction"]}/{cf["mask"]["cone"]["N"]}'
        r["base_mae_cf"] = cf["base"]["MAE"]
    # training-log metrics at the best epoch (val, standard accumulator) - v10 only
    m = load(CHECKPOINT_DIR / tag / "metrics.json")
    if m and m.get("composite"):
        best = int(np.argmax(m["composite"]))
        r["best_epoch"] = best + 1
        r["mAP_50"] = m["mAP_50"][best]
        r["obs_f1"] = m["obs_f1"][best]
    # cls_f1 (macro, instance level) for EVERY model from the threshold sweep
    # at the current OBJ_THRESH - same code path for model 1 and v10
    td = load(RESULTS_DIR / ("decode_tuning_val.json" if tag == MODEL1 else f"decode_tuning_val_{tag}.json"))
    if td:
        thr = str(td["current"]["OBJ_THRESH"])
        sw = td["obj_thresh_sweep"].get(thr)
        if sw:
            r["cls_f1"] = sw["macro_F1"]
            r["per_class_F1"] = {k: v["F1"] for k, v in sw["per_class"].items()}
    return r


def fmt(v, f="{:.2f}"):
    return f.format(v) if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) else "  -  "


def main():
    rows = [row(MODEL1)] + [row(t) for t in V10]
    print(f'{"":<16}{"cls_f1":>8}{"mAP":>7}{"obs_f1":>8}{"MAE":>7}{"N":>6}{"lookup":>8}{"slope":>8}{"slope_y":>8}{"shear":>7}{"spur":>7}{"shift+":>8}{"cone-hid":>10}')
    for r in rows:
        print(
            f'{r["tag"]:<16}{fmt(r.get("cls_f1")):>8}{fmt(r.get("mAP_50")):>7}{fmt(r.get("obs_f1")):>8}'
            f'{fmt(r.get("angle_mae")):>7}{fmt(r.get("angle_n"), "{:d}"):>6}{fmt(r.get("lookup_baseline")):>8}'
            f'{fmt(r.get("stretch_slope")):>8}{fmt(r.get("slope_y")):>8}{fmt(r.get("slope_shear")):>7}{fmt(r.get("spurious")):>7}{fmt(r.get("shift_increase")):>8}{r.get("mask_cone_nopred", "-"):>10}'
        )

    m1 = rows[0]
    verdicts = {}
    for r in rows[1:]:
        v = {}
        if "cls_f1" in r and "cls_f1" in m1:
            v["1_cls_f1_up"] = r["cls_f1"] > m1["cls_f1"]
        else:
            v["1_cls_f1_up"] = "macro F1 missing: run tune_decode.py for both models"
        if "angle_mae" in r and "angle_mae" in m1:
            v["2_mae_not_worse_1deg"] = r["angle_mae"] <= m1["angle_mae"] + 1.0
        if "shift_increase" in r and "shift_increase" in m1:
            v["3_shift_not_worse"] = r["shift_increase"] <= m1["shift_increase"]
        if "stretch_slope" in r and "stretch_slope" in m1:
            v["4_stretch_slope_up"] = r["stretch_slope"] > m1["stretch_slope"]
        verdicts[r["tag"]] = v
        print(f'\n{r["tag"]}:')
        for k, val in v.items():
            print(f"   {k:<24} {val}")

    out = {"rows": rows, "verdicts": verdicts, "rules": "see docs/notes/PLAN-V10.md section 6"}
    name = "compare_1e.json" if not sys.argv[1:] else "compare_" + "_".join(V10) + ".json"
    (RESULTS_DIR / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWritten: {RESULTS_DIR / name}")


if __name__ == "__main__":
    main()
