"""
common.py - shared helpers for model_2/evaluation scripts.

Every evaluation script imports the model code from ../structure so that
model_2 is self-contained and never touches train_v9_v1 (model 1).
"""

import sys
from pathlib import Path

import torch

EVAL_DIR = Path(__file__).resolve().parent
MODEL2_DIR = EVAL_DIR.parent
STRUCTURE_DIR = MODEL2_DIR / "structure"
RESULTS_DIR = MODEL2_DIR / "results"
CHECKPOINT_DIR = MODEL2_DIR / "checkpoint"

if str(STRUCTURE_DIR) not in sys.path:
    sys.path.insert(0, str(STRUCTURE_DIR))

from metrics import IOU_MATCH_THRESH  # noqa: E402,F401  (single source of truth)
from model import AFGNet  # noqa: E402

from config import REPO_ROOT  # noqa: E402

# Model 1 (v9) reference checkpoint - the baseline every model_2 run is compared
# against. NOTE: it was trained at 640x640 / rho (2, 3,20,20); to score it run
# the evaluation with AFGNET_V9_MODE=1 so the grid and ranges match.
MODEL1_CKPT = REPO_ROOT / "train" / "model_1" / "checkpoint" / "v9_fixes_reverse_seed42" / "train_v9_v1.pth"


def load_model(ckpt_path: Path, device: torch.device) -> AFGNet:
    """Load a checkpoint into the model_2/structure AFGNet.

    cell_x / cell_y are deterministic buffers computed from FEAT_H/FEAT_W;
    they are dropped from the state dict so a checkpoint trained at one grid
    size does not clash with the freshly built model.
    """
    model = AFGNet(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    state.pop("head.cell_x", None)
    state.pop("head.cell_y", None)
    res = model.load_state_dict(state, strict=False)
    missing = [k for k in res.missing_keys if k not in ("head.cell_x", "head.cell_y", "head.kpt_rho")]
    if missing:
        raise RuntimeError(f"Missing learned keys: {missing}")
    model.to(device).eval()
    if isinstance(ckpt, dict):
        print(f'    checkpoint: epoch={ckpt.get("epoch", "?")}, val_composite={ckpt.get("composite", float("nan")):.4f}')
    return model
