"""
evidence_hparams.py - measure, from the annotations alone, what the two
cell-aware regression ranges must be.

No training, no checkpoint, no GPU.  Every number printed here is derived
from the split JSONs plus the assignment rule in targets.py, so it can be
recomputed by anyone and quoted in the thesis.

Background
----------
model.py decodes a box centre as

    pred_cx = cell_centre_x + tanh(dx) * cell_w * REGRESSION_RANGE

so a cell can only place a box centre within REGRESSION_RANGE cells of
itself.  targets.py makes EVERY cell whose centre falls inside a GT box a
positive cell for that box.  If a positive cell sits further from the box
centre than REGRESSION_RANGE, that cell cannot output its own target, and
its box loss has a floor above zero.

The same applies per keypoint through KPT_RHO_PER_KPT.

Outputs results/hparam_evidence.json
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from common import RESULTS_DIR  # also puts ../structure on sys.path
from dataset import AFGNetDataset
from model import KPT_RHO_PER_KPT, REGRESSION_RANGE

from config import (
    CLASS_NAMES,
    FEAT_H,
    FEAT_W,
    REPO_ROOT,
    SPRAY_IDX,
    TEST_JSON,
    TRAIN_JSON,
)

KPT_NAMES = ["tip", "left_edge", "right_edge"]
RHO_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 20]


def assign_positive_cells(gt_boxes):
    """Replicate the assignment in targets.py. Returns (i, j, gt_idx) triples."""
    assigned_area = np.full((FEAT_H, FEAT_W), 1e9, dtype=np.float32)
    assigned_idx = np.full((FEAT_H, FEAT_W), -1, dtype=np.int32)

    for idx, (cx, cy, bw, bh) in enumerate(gt_boxes):
        if bw <= 0 or bh <= 0:
            continue
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        j_min = max(0, math.floor(x1 * FEAT_W))
        j_max = min(FEAT_W - 1, math.ceil(x2 * FEAT_W - 1))
        i_min = max(0, math.floor(y1 * FEAT_H))
        i_max = min(FEAT_H - 1, math.ceil(y2 * FEAT_H - 1))
        area = bw * bh

        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                ccx = (j + 0.5) / FEAT_W
                ccy = (i + 0.5) / FEAT_H
                if not (x1 <= ccx <= x2 and y1 <= ccy <= y2):
                    continue
                if area >= assigned_area[i, j]:
                    continue
                assigned_area[i, j] = area
                assigned_idx[i, j] = idx

    out = []
    for i, j in zip(*np.nonzero(assigned_idx >= 0)):
        out.append((int(i), int(j), int(assigned_idx[i, j])))
    return out


def pct(a, qs=(50, 90, 95, 99, 99.9, 100)):
    if len(a) == 0:
        return {}
    return {"p%g" % q: float(np.percentile(a, q)) for q in qs}


def measure(split_json, label):
    ds = AFGNetDataset(split_json, augment=False)
    print(
        "  %s: %d images loaded (skipped %d missing, %d empty)"
        % (label, len(ds.samples), ds.skipped_no_image, ds.skipped_no_anns)
    )

    box_reach = []  # per positive cell: distance to own box centre, in cells
    box_reach_cls = {c: [] for c in range(len(CLASS_NAMES))}
    box_w_cells, box_h_cells = [], []
    kpt_reach = [[] for _ in KPT_NAMES]  # spray cells only
    cells_per_box = Counter()
    spray_centres = []

    for s in ds.samples:
        boxes = s["gt_boxes"]
        classes = s["gt_classes"]
        kpts = s["gt_kpts"]

        for (cx, cy, bw, bh), c in zip(boxes, classes):
            box_w_cells.append(bw * FEAT_W)
            box_h_cells.append(bh * FEAT_H)
            if c == SPRAY_IDX:
                spray_centres.append((round(cx, 4), round(cy, 4)))

        per_box = Counter()
        for i, j, gi in assign_positive_cells(boxes):
            ccx = (j + 0.5) / FEAT_W
            ccy = (i + 0.5) / FEAT_H
            cx, cy, bw, bh = boxes[gi]
            d = max(abs(cx - ccx) * FEAT_W, abs(cy - ccy) * FEAT_H)
            box_reach.append(d)
            box_reach_cls[classes[gi]].append(d)
            per_box[gi] += 1

            if classes[gi] == SPRAY_IDX:
                kp = kpts[gi]
                if any(v > 0 for v in kp):
                    for k in range(len(KPT_NAMES)):
                        kx, ky = kp[2 * k], kp[2 * k + 1]
                        dk = max(abs(kx - ccx) * FEAT_W, abs(ky - ccy) * FEAT_H)
                        kpt_reach[k].append(dk)
        cells_per_box.update(per_box.values())

    box_reach = np.asarray(box_reach)
    res = {
        "split": str(split_json),
        "n_images": len(ds.samples),
        "n_positive_cells": int(box_reach.size),
        "cells_per_box": pct(
            np.asarray(list(cells_per_box.elements()) or [0], dtype=float)
        ),
        "box_size_cells": {
            "w": pct(np.asarray(box_w_cells)),
            "h": pct(np.asarray(box_h_cells)),
        },
        "box_centre_reach_cells": pct(box_reach),
        "box_unreachable_fraction": {
            str(r): float((box_reach > r).mean()) for r in RHO_GRID
        },
        "box_unreachable_fraction_per_class": {
            CLASS_NAMES[c]: {
                str(r): (float((np.asarray(v) > r).mean()) if v else None)
                for r in RHO_GRID
            }
            for c, v in box_reach_cls.items()
        },
        "keypoint_reach_cells": {
            KPT_NAMES[k]: dict(
                {"n": len(kpt_reach[k])},
                **pct(np.asarray(kpt_reach[k])),
                **{
                    "unreachable_fraction": {
                        str(r): (
                            float((np.asarray(kpt_reach[k]) > r).mean())
                            if kpt_reach[k]
                            else None
                        )
                        for r in RHO_GRID
                    }
                }
            )
            for k in range(len(KPT_NAMES))
        },
        "distinct_spray_box_centres": len(set(spray_centres)),
        "n_spray_boxes": len(spray_centres),
    }
    return res


def reachable_area_fraction(rho):
    """Mean fraction of the image a single cell can reach at this rho.

    A cell at (i, j) reaches [j-rho, j+rho] x [i-rho, i+rho] cells, clipped
    to the map, averaged over all cells.  1.0 means the constraint never
    binds anywhere - the cell-aware anchor then carries no information.
    """
    js = np.arange(FEAT_W) + 0.5
    is_ = np.arange(FEAT_H) + 0.5
    w = np.minimum(js + rho, FEAT_W) - np.maximum(js - rho, 0)
    h = np.minimum(is_ + rho, FEAT_H) - np.maximum(is_ - rho, 0)
    return float(np.outer(h, w).mean() / (FEAT_H * FEAT_W))


def main():
    print("=" * 70)
    print("Hyper-parameter evidence - derived from annotations, not from results")
    print("=" * 70)
    print(
        "Feature map %dx%d, current REGRESSION_RANGE=%s, KPT_RHO_PER_KPT=%s"
        % (FEAT_H, FEAT_W, REGRESSION_RANGE, KPT_RHO_PER_KPT)
    )
    print()

    out = {
        "_meta": {
            "feat": [FEAT_H, FEAT_W],
            "current_rho_box": REGRESSION_RANGE,
            "current_rho_kpt": list(KPT_RHO_PER_KPT),
        },
        "reachable_image_fraction_per_rho": {
            str(r): reachable_area_fraction(r) for r in RHO_GRID
        },
    }

    for label, js in (("train", TRAIN_JSON), ("test", TEST_JSON)):
        print("[%s]" % label)
        out[label] = measure(js, label)

    dst = RESULTS_DIR / "hparam_evidence.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")

    tr = out["train"]
    print()
    print("-" * 70)
    print("Positive cells (train): %d" % tr["n_positive_cells"])
    print()
    print("BOX CENTRE - how far a positive cell sits from its own box centre")
    print("  percentiles (cells): %s" % tr["box_centre_reach_cells"])
    print("  fraction of positive cells that CANNOT reach their target:")
    for r in RHO_GRID:
        f = tr["box_unreachable_fraction"][str(r)]
        mark = "   <-- current" if r == REGRESSION_RANGE else ""
        print("    rho=%-3d %6.2f%%%s" % (r, f * 100, mark))
    print()
    print("KEYPOINTS - distance from the spray cell to each keypoint")
    for k in KPT_NAMES:
        d = tr["keypoint_reach_cells"][k]
        print(
            "  %-11s n=%-7d p50=%.2f p99=%.2f max=%.2f cells"
            % (k, d["n"], d.get("p50", 0), d.get("p99", 0), d.get("p100", 0))
        )
    print()
    print("Fraction of the image one cell can reach, per rho:")
    for r in RHO_GRID:
        print(
            "    rho=%-3d %6.2f%% of image"
            % (r, out["reachable_image_fraction_per_rho"][str(r)] * 100)
        )
    print()
    print(
        "Distinct spray box centres in train: %d out of %d spray boxes"
        % (tr["distinct_spray_box_centres"], tr["n_spray_boxes"])
    )
    print()
    print("Written: %s" % dst)


if __name__ == "__main__":
    main()
