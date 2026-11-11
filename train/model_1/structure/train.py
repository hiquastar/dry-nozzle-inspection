"""
train.py - AFGNet v9 main training

Usage:
    python train.py             # full 100 epochs
    python train.py --debug     # 1 epoch on subset
    python train.py --epochs 50 # custom epochs
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Force UTF-8 stdout (Windows fix for ★ emoji)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dataset import AFGNetDataset, afgnet_collate
from loss import AFGNetLoss
from metrics import MetricsAccumulator
from model import AFGNet, count_parameters

from config import (
    BATCH_SIZE,
    DEVICE,
    LR,
    METRICS_LOG,
    MODEL_PATH,
    NUM_WORKERS,
    RESUME_FROM,
    RUN_DIR,
    SEED,
    TOTAL_EPOCHS,
    TRAIN_JSON,
    VAL_JSON,
    WEIGHT_DECAY,
    print_config,
)


def set_seed(seed: int) -> torch.Generator:
    """Seed every RNG the training loop touches and make cuDNN deterministic.

    Before 2026-09-16 no seed was ever applied (SEED was imported but unused),
    so runs named *_seed42 were not reproducible. Two runs with the same seed
    must now produce the same per-epoch losses; that identity is the gate
    used to validate every later patch (see docs/notes/PLAN-V10.md).

    Returns the generator to hand to DataLoader so worker shuffling is seeded too.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: a few backward kernels have no deterministic variant; report
    # them instead of aborting, so the gate test shows how exact we can be.
    torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init: derive per-worker seeds from torch's base seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def run_epoch(model, loader, loss_fn, optimizer, is_train, desc, scaler=None):
    """One epoch loop. Returns (avg_loss, breakdown_avg, metrics_dict)."""
    model.train() if is_train else model.eval()
    total_loss = 0.0
    parts = defaultdict(float)
    accumulator = MetricsAccumulator()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")

    n_batches = 0
    with ctx:
        for imgs, tgts in bar:
            imgs = imgs.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                out = model(imgs)

            # Loss outside autocast (BCE not safe in fp16)
            loss, lp = loss_fn(out, tgts, DEVICE)

            if not torch.isfinite(loss):
                bar.write(f"  [SKIP] non-finite loss={loss.item()}")
                continue

            if is_train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()

            total_loss += loss.item()
            for k, v in lp.items():
                if isinstance(v, (int, float)):
                    parts[k] += v
            n_batches += 1

            # Accumulate metrics for validation
            if not is_train:
                accumulator.update_from_batch(out, tgts, DEVICE)

            bar.set_postfix(loss=f"{loss.item():.4f}")

    n = max(n_batches, 1)
    avg_loss = total_loss / n
    avg_parts = {k: v / n for k, v in parts.items()}

    metrics = {}
    if not is_train:
        metrics = accumulator.compute_summary()

    return avg_loss, avg_parts, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="1 epoch on small subset")
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--resume", type=str, default=RESUME_FROM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Write outputs to checkpoint/<run_name>/ instead of the config RUN_DIR "
        "(use for gate tests and experiments so the reference checkpoint is never overwritten)",
    )
    args = parser.parse_args()

    global RUN_DIR, MODEL_PATH, METRICS_LOG
    if args.run_name:
        RUN_DIR = RUN_DIR.parent / args.run_name
        MODEL_PATH = RUN_DIR / MODEL_PATH.name
        METRICS_LOG = RUN_DIR / METRICS_LOG.name

    loader_gen = set_seed(args.seed)

    print_config()
    print(f"Seed:         {args.seed} (deterministic cuDNN)")
    print(f"\nRun dir: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────
    # Data (load COCO splits directly)
    # ─────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = AFGNetDataset(TRAIN_JSON, augment=True)
    train_ds.print_stats()

    val_ds = AFGNetDataset(VAL_JSON, augment=False)
    val_ds.print_stats()

    if args.debug:
        from torch.utils.data import Subset

        train_ds = Subset(train_ds, list(range(min(50, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(20, len(val_ds)))))
        args.epochs = 1
        print(
            f"\n[DEBUG MODE] Using {len(train_ds)} train + {len(val_ds)} val samples, 1 epoch"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=afgnet_collate,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=afgnet_collate,
        pin_memory=True,
    )
    print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ─────────────────────────────────────
    # Model
    # ─────────────────────────────────────
    print("\nBuilding model...")
    model = AFGNet(pretrained=True).to(DEVICE)

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            state = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(state, strict=False)
            print(f"Resumed from: {ckpt_path}")

    total, trainable = count_parameters(model)
    print(f"Total params: {total:,}, Trainable: {trainable:,}")

    # ─────────────────────────────────────
    # Optimizer + scheduler + loss
    # ─────────────────────────────────────
    loss_fn = AFGNetLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    # ─────────────────────────────────────
    # Train
    # ─────────────────────────────────────
    print(f'\n{"=" * 60}')
    print(f"Starting training: {args.epochs} epochs")
    print(f'{"=" * 60}\n')

    history = defaultdict(list)
    best_composite = 0.0

    for epoch in range(args.epochs):
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs}  (LR={cur_lr:.2e})")

        # Train
        tr_loss, tr_parts, _ = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            is_train=True,
            desc=f"Train E{epoch+1}",
            scaler=scaler,
        )

        # Val
        va_loss, va_parts, va_metrics = run_epoch(
            model,
            val_loader,
            loss_fn,
            optimizer,
            is_train=False,
            desc=f"Val E{epoch+1}",
            scaler=None,
        )

        scheduler.step()

        # Composite
        _map50 = va_metrics.get("mAP_50", 0.0)
        _f1 = va_metrics.get("cls_f1", 0.0)
        _obs_f1 = va_metrics.get("obs_f1", 0.0)
        _a_mae = va_metrics.get("angle_mae_deg", 10.0)
        if not isinstance(_a_mae, float) or math.isnan(_a_mae):
            _a_mae = 10.0
        _angle_score = max(0.0, 1.0 - _a_mae / 10.0)
        _composite = 0.30 * _map50 + 0.25 * _f1 + 0.20 * _angle_score + 0.25 * _obs_f1

        # Log
        print(f"  train_loss: {tr_loss:.4f}, val_loss: {va_loss:.4f}")
        print(
            f"  mAP@.5={_map50:.3f}  cls_f1={_f1:.3f}  obs_f1={_obs_f1:.3f}  "
            f"angle_mae={_a_mae:.2f}°  composite={_composite:.4f}"
        )

        targets_met = []
        if _map50 >= 0.85:
            targets_met.append("mAP")
        if _f1 >= 0.90:
            targets_met.append("cls")
        if _obs_f1 >= 0.80:
            targets_met.append("obs")
        if _a_mae <= 5.0:
            targets_met.append("angle")
        if targets_met:
            print(f'  Targets met: {", ".join(t + chr(10003) for t in targets_met)}')

        # Save best
        if _composite > best_composite:
            best_composite = _composite
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "composite": _composite,
                    "metrics": va_metrics,
                },
                MODEL_PATH,
            )
            print(f"  \u2605 New best! Saved to {MODEL_PATH.name}")

        # History
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["mAP_50"].append(_map50)
        history["cls_f1"].append(_f1)
        history["obs_f1"].append(_obs_f1)
        history["angle_mae"].append(_a_mae)
        history["composite"].append(_composite)
        history["lr"].append(cur_lr)

        # Save metrics log every epoch
        with open(METRICS_LOG, "w") as f:
            json.dump(dict(history), f, indent=2)

    # ─────────────────────────────────────
    # Done
    # ─────────────────────────────────────
    print(f'\n{"=" * 60}')
    print(f"Training complete. Best composite score: {best_composite:.4f}")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics log: {METRICS_LOG}")
    print(f'{"=" * 60}')


if __name__ == "__main__":
    main()
