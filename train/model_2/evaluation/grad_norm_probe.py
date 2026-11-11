"""
grad_norm_probe.py - how much of the training gradient does each loss term
actually contribute?

The loss weights (obj 1, box 8, cls 2, kpt 30, obs 2, iou 2) have no
recorded derivation except kpt. A weight only means something through the
gradient it sends into the SHARED trunk (backbone + FPN + head stem), so
this script measures exactly that, per term, on real training batches:

    raw_k      = || d L_k / d theta_shared ||      (weight = 1)
    weighted_k = w_k * raw_k                        (what training sees)
    share_k    = weighted_k / sum_j weighted_j

It also prints the weights that WOULD equalise the shares (w_k ~ 1/raw_k,
scaled so obj = 1). That is a derived suggestion for an ablation, not a
change applied anywhere.

One forward per batch, one backward per term (retain_graph), model in
train() mode so BatchNorm/Dropout behave as in training; seeded so the
numbers are reproducible.

    python train/model_2/evaluation/grad_norm_probe.py                 # v10 default ckpt
    AFGNET_V9_MODE=1 python .../grad_norm_probe.py --ckpt train/model_1/checkpoint/v9_fixes_reverse_seed42/train_v9_v1.pth

Output: results/grad_norm_<tag>.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import MODEL1_CKPT, RESULTS_DIR, load_model  # also puts ../structure on sys.path
from dataset import AFGNetDataset, afgnet_collate
from loss import AFGNetLoss

from config import ANGLE_LOSS, BATCH_SIZE, DEVICE, LOSS_WEIGHTS, MIST_HEAD, TRAIN_JSON, VAL_JSON

TERMS = ["obj", "box", "cls", "kpt", "obs", "iou"] + (["angle"] if ANGLE_LOSS else []) + (["mist"] if MIST_HEAD else [])


def shared_params(model):
    """Trunk parameters: everything except the per-task 1x1 output convs."""
    heads = ("head.det_head", "head.iou_head", "head.cls_head", "head.kpt_head")
    return [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(heads)]


def grad_norm(params):
    return float(torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in params if p.grad is not None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(MODEL1_CKPT))
    ap.add_argument("--label", default=None)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--no_augment", action="store_true", help="disable train-time augmentation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", action="store_true", help="probe an untrained model (ImageNet backbone, random heads) instead of --ckpt")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    ckpt = Path(args.ckpt)
    tag = args.label or ("init" if args.init else ckpt.parent.name)
    ds = AFGNetDataset(TRAIN_JSON if args.split == "train" else VAL_JSON, augment=(args.split == "train" and not args.no_augment))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=afgnet_collate, drop_last=True, generator=g)

    if args.init:
        from model import AFGNet

        model = AFGNet(pretrained=True).to(DEVICE)
        ckpt = Path("(untrained: ImageNet backbone, random heads)")
    else:
        model = load_model(ckpt, DEVICE)
    model.train()  # gradients as during training (BN batch stats, dropout on)
    params = shared_params(model)
    loss_fn = AFGNetLoss().to(DEVICE)

    raw = {k: [] for k in TERMS}
    vals = {k: [] for k in TERMS}
    it = iter(loader)
    for _ in tqdm(range(args.batches), desc="  batches", unit="batch"):
        imgs, tgts = next(it)
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        for k in TERMS:
            loss_fn.lw = {t: (1.0 if t == k else 0.0) for t in TERMS}  # isolate one term, weight 1
            total, parts = loss_fn(out, tgts, DEVICE)
            model.zero_grad(set_to_none=True)
            if total.requires_grad:
                total.backward(retain_graph=True)
                raw[k].append(grad_norm(params))
            else:
                raw[k].append(0.0)
            vals[k].append(float(parts[k]))
        model.zero_grad(set_to_none=True)
        del out

    loss_fn.lw = LOSS_WEIGHTS
    raw_m = {k: float(np.mean(raw[k])) for k in TERMS}
    weighted = {k: LOSS_WEIGHTS[k] * raw_m[k] for k in TERMS}
    tot = sum(weighted.values())
    share = {k: weighted[k] / tot if tot > 0 else float("nan") for k in TERMS}
    # weights that would give every term the same trunk gradient, obj fixed at 1
    eq = {k: (raw_m["obj"] / raw_m[k]) if raw_m[k] > 0 else float("nan") for k in TERMS}

    res = {
        "checkpoint": str(ckpt),
        "split": args.split,
        "augment": bool(ds.augment),
        "batches": args.batches,
        "batch_size": BATCH_SIZE,
        "current_loss_weights": LOSS_WEIGHTS,
        "loss_value_mean": {k: float(np.mean(vals[k])) for k in TERMS},
        "grad_norm_raw_mean": raw_m,
        "grad_norm_raw_std": {k: float(np.std(raw[k])) for k in TERMS},
        "grad_norm_weighted": weighted,
        "share_of_trunk_gradient": share,
        "equalising_weights_suggestion": eq,
        "_note": "shares are of the sum of per-term weighted gradient norms on the shared trunk; equalising weights are a derived suggestion for an ablation, not applied",
    }
    out = RESULTS_DIR / f"grad_norm_{tag}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\n{tag}  ({args.split}, augment={ds.augment}, {args.batches} batches x {BATCH_SIZE})")
    print(f'{"term":<5}{"weight":>8}{"loss":>9}{"raw |grad|":>12}{"weighted":>11}{"share":>8}{"equalising w":>14}')
    for k in TERMS:
        print(f'{k:<5}{LOSS_WEIGHTS[k]:>8.1f}{res["loss_value_mean"][k]:>9.4f}{raw_m[k]:>12.4f}{weighted[k]:>11.4f}{share[k]*100:>7.1f}%{eq[k]:>14.2f}')
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
