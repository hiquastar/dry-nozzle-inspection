# model_2 — AFGNet v10

Kept separate from model 1 ([`train/model_1`](../model_1/README.md), with checkpoints in `train/model_1/checkpoint/`). All v10
changes are made here; `train_v9_v1` is no longer modified.

```
model_2/
├── structure/     model code: config, dataset, targets, loss, model, metrics, train
├── evaluation/    new evaluation suite (scripts)
├── results/       output of the evaluation scripts
├── deploy/        ONNX export, parity check, Pi 5 packaging
├── data/          splits used for v10 (a copy of data/splits_reverse)
└── checkpoint/    v10 runs (train.py --run_name writes here)
```

## structure/

This started as a copy of `train_v9_v1` **after step 0** (angle computed in the correct space, a real seed,
`--seed`, `--run_name`). The paths in `config.py` point to `model_2/data` and `model_2/checkpoint`.

**Step 1 (2026-09-16) — four flags in `config.py`, all switched on for v10:**

| Flag | What it changes | File |
|---|---|---|
| `NEW_RHO` | box ρ 2 → 10, keypoint ρ (3,20,20) → (9,14,14) | `config.py` → `model.py` |
| `LETTERBOX` | 640×480 input (keeps 4:3), 40×30 grid | `config.py` → `dataset.py`, `model.py` |
| `GEO_AUG` | rotation ±15°, shift ±10%, isotropic scale 0.8–1.2, applied in pixels, with an assert | `dataset.py: random_geo_affine` |
| `INSTANCE_NORM` | each cell is weighted by 1/(number of cells in its box) → every nozzle counts equally | `targets.py: inst_weight` → `loss.py: reduce` |

`AFGNET_V9_MODE=1` switches all four off and reproduces the v9 code path exactly (verified: a debug run gives
`train_loss 9.4160 / val_loss 14.5996`, the same as the old code). Use this mode for the **regression check** and
to score model 1 checkpoints with the new evaluation suite.

**The old evaluation has been removed from `metrics.py` entirely.** There are no cell-level metrics any more. `MetricsAccumulator`
reads the raw GT through `targets["index"]` and computes: class-aware mAP, per-class F1 at **instance level** (macro),
obs F1 on spray/facing instances matched class-agnostically, and angle MAE **matched by apex** (≤ 50 px, decoded at 0.05)
in true degrees. `train.py` uses exactly these numbers to pick the checkpoint.

Run: `python structure/train.py --run_name v10_ref_s42 --seed 42` (writes to `checkpoint/v10_ref_s42/`).

## evaluation/

| Script | Question it answers | Output |
|---|---|---|
| `evidence_hparams.py` | what ρ should be — measured from the labels, no checkpoint needed | `results/hparam_evidence.json` |
| `eval_angle_reeval.py` | angle MAE (true degrees) + two trivial baselines + CI | `results/angle_reeval.json` |
| `tune_decode.py` | detection threshold and keypoint merging — tuned on val | `results/decode_tuning_val.json` |
| `diag_missed_sprays.py` | why 1/3 of the cones are "not detected" | `results/diag_missed_sprays.json` |
| `eval_counterfactual.py` | does the network **measure** or **memorise** — stretch / mask / shift / jitter | `results/counterfactual_<split>_<run>.json` |
| `common.py` | `load_model` and shared paths | — |

You can run them from any directory: `python train/model_2/evaluation/<script>.py`. The default checkpoint is
model 1 (`train/model_1/checkpoint/v9_fixes_reverse_seed42`), used as the baseline; pass `--ckpt` to evaluate a v10 run.

Rule: tune parameters on **val** only; run **test** once, on the final model.
