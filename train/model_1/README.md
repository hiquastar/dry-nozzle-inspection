# model_1 — AFGNet v9

The thesis's earlier model (before the mist head and the angle reader were added). It is kept as a baseline for
model 2 and so the live app can still run v9. It has the same layout as [model_2](../model_2/README.md).

```
model_1/
├── structure/     model code: config, dataset, targets, loss, model, metrics, train
├── evaluation/    evaluation, confusion matrix, benchmark, plots (scripts)
├── results/       output of the evaluation scripts
├── deploy/        export_onnx.py
├── checkpoint/    v9 runs (train.py --run_name writes here)
└── train_v9_v1.rar  archive of the original code
```

## Data

Uses the reverse split in `data/splits_reverse/` (CAM2 for train, CAM1 divided into val/test). Its content is
identical to `train/model_2/data/`. Images are read from `data/images/`.

## checkpoint/

| Run | Contents |
|---|---|
| `v9_fixes_reverse_seed42/` | The thesis run: `train_v9_v1.pth`, `metrics.json`, `test_results.json`, ONNX and the Pi 5 package (`pi5_package/`, `pi5_benchmark_v2/`) |
| `v9_rabl_A_*`, `v9_rabl_B_*`, `v9_rabl_C_*`, `v9_abl_D_*` | Ablations |
| `Checkpoint_v/` | Snapshots of older `model.py` versions |

New training runs write to `checkpoint/v9_ref/` by default (or `checkpoint/<run_name>/`), so they never overwrite
the thesis run. New `.pth` files are not committed (see `.gitignore`).

## results/

| File | Contents |
|---|---|
| `chapter6_final_numbers.json` | v9 test-set numbers (`eval_test_set.py`, `eval_all_runs.py`, `eval_mae_split.py`) |
| `figures/` | Confusion matrices and detection tables (`make_confusion_matrices.py`) |
| `infer/` | Inference videos at several thresholds (`video_infer.py`) |

## Running

```bash
cd train/model_1/structure && python train.py --run_name my_run      # train
cd train/model_1/evaluation && python eval_test_set.py                # test set
cd train/model_1/deploy && python export_onnx.py                      # ONNX
```
