"""
map_gate_probe.py - the class-aware mAP@0.5 of the REPORTED state, i.e. after
the mist gate has relabelled every small non-spray detection, next to the mAP
of the classification head that map_threshold_probe.py computes.

Detections are decoded at the same low threshold as the head mAP (0.05), so
the two numbers differ only in the class label: the head gives spray / big as
before, and the gate decides no_spray versus facing from the mist at the tip,
with the threshold that eval_mist_hybrid calibrated on the training set.

    AFGNET_INSTANCE_NORM=0 AFGNET_TIP_RELATIVE_EDGES=1 AFGNET_MIST_HEAD=1 AFGNET_LEARNED_HEADS=1 \
      python evaluation/map_gate_probe.py --split test --seed 42

Output: results/map_gate_<split>_final_heads_s<seed>.json
"""

import argparse
import json

from tqdm import tqdm

import eval_mist_hybrid as MH
from common import RESULTS_DIR, load_model
from dataset import AFGNetDataset, load_frame
from metrics import compute_map_classaware, decode_predictions

from config import CLASS_NAMES, DEVICE, FACING_IDX, NO_SPRAY_IDX, REPO_ROOT, SPRAY_IDX, TEST_JSON, VAL_JSON

SPLITS = {"val": VAL_JSON, "test": TEST_JSON}
DECODE_THR = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default="test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tag = f"final_heads_s{args.seed}"
    ckpt = REPO_ROOT / "train" / "model_2" / "checkpoint" / f"abl_heads_s{args.seed}" / "afgnet_v10.pth"
    thr = json.load(open(RESULTS_DIR / f"mist_hybrid_{args.split}_{tag}.json"))["state"]["gate_threshold_from_train"]
    model = load_model(ckpt, DEVICE)
    ds = AFGNetDataset(SPLITS[args.split], augment=False)

    preds_head, preds_gate, gts, gcls = [], [], [], []
    for s in tqdm(ds.samples, desc=f"seed {args.seed} on {args.split}", unit="frame"):
        prob, out = MH.predict_map(model, ds, load_frame(s["image_path"]))
        dets = decode_predictions(out, batch_idx=0, obj_thresh=DECODE_THR)
        gate = []
        for d in dets:
            g = dict(d)
            if d["class_id"] in (NO_SPRAY_IDX, FACING_IDX):
                g["class_id"] = FACING_IDX if MH.tip_disk_mean(prob, d["box"]) > thr else NO_SPRAY_IDX
            gate.append(g)
        preds_head.append(dets)
        preds_gate.append(gate)
        gts.append(s["gt_boxes"])
        gcls.append(s["gt_classes"])

    res = {"checkpoint": str(ckpt), "split": args.split, "decode_threshold": DECODE_THR, "gate_threshold": thr}
    for name, preds in (("classification_head", preds_head), ("reported_state", preds_gate)):
        m = compute_map_classaware(preds, gts, gcls)
        res[name] = {"mAP_50": m["mAP"], "per_class": dict(zip(CLASS_NAMES, m["per_class"]))}
        pc = "  ".join(f"{c.replace('nozzle_', '').replace('_nozzle', '')} {v:.3f}" for c, v in res[name]["per_class"].items())
        print(f"  {name:<20} mAP@0.5 {m['mAP']:.3f}   [{pc}]")
    out = RESULTS_DIR / f"map_gate_{args.split}_{tag}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("Written:", out)


if __name__ == "__main__":
    main()
