"""
config.py - AFGNet v9 configuration

PATCHED (May 11): LOSS_WEIGHTS['kpt'] raised from 10 -> 30 to compensate
for the smaller numeric scale of cell-aware keypoint offsets (offsets are
~0.05 vs absolute coords ~0.5, so smooth_l1 gradients are ~10x smaller).

AFGNet v9 = cxcywh detector + hierarchical conditional masking (thesis novelty).
"""

from datetime import datetime
from pathlib import Path

import torch

# ============================================================================
# PATHS
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parents[3]  # structure -> model_1 -> train -> repo root
MODEL1_ROOT = REPO_ROOT / "train" / "model_1"

# Reverse split: CAM2 in train, CAM1 split into val/test (for fixes retrain).
SPLITS_ROOT = REPO_ROOT / "data" / "splits_reverse"
TRAIN_JSON = SPLITS_ROOT / "train.coco.json"
VAL_JSON = SPLITS_ROOT / "val.coco.json"
TEST_JSON = SPLITS_ROOT / "test.coco.json"

# Image roots (where images live)
ORIGINAL_IMG_ROOT = REPO_ROOT / "data" / "images"
CONVERTED_IMG_ROOT = REPO_ROOT / "data" / "annotations" / "v2"

# Run output. Every v9 run lives under model_1/checkpoint/<name> (like model_2);
# pass --run_name to train.py for anything else. The thesis run is
# checkpoint/v9_fixes_reverse_seed42, so a new run never overwrites it.
VERSION = "v9_ref"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = MODEL1_ROOT / "checkpoint" / VERSION
TRAIN_VERSION = "train_v9_v1"  # checkpoint file name used by every v9 script
MODEL_PATH = RUN_DIR / f"{TRAIN_VERSION}.pth"
ONNX_PATH = RUN_DIR / f"{TRAIN_VERSION}.onnx"
METRICS_LOG = RUN_DIR / "metrics.json"

# Resume from checkpoint (None = train from ImageNet pretrained)
RESUME_FROM = None


# ============================================================================
# IMAGE / TRAINING HYPERPARAMETERS
# ============================================================================
IMG_SIZE = 640
BATCH_SIZE = 16
NUM_WORKERS = 4
SEED = 42
VAL_RATIO = 0.25

LR = 1e-4
TOTAL_EPOCHS = 100
WEIGHT_DECAY = 5e-4

# Source frame geometry. Every frame in the corpus is 2048x1536 (4:3).
# Keypoints are stored normalised by W and H separately, so the normalised
# plane is an anisotropic squash of the pixel plane. Any angle must be
# computed after undoing that squash (x * ANGLE_ASPECT), otherwise the
# reported degrees are not physical degrees (measured error: -5..+11 deg
# depending on cone orientation). dataset.py warns if a frame deviates.
ORIG_IMG_W = 2048
ORIG_IMG_H = 1536
ANGLE_ASPECT = ORIG_IMG_W / ORIG_IMG_H

# Feature map (FCOS detection)
FEAT_STRIDE = 16
FEAT_H = IMG_SIZE // FEAT_STRIDE
FEAT_W = IMG_SIZE // FEAT_STRIDE
FPN_CHANNELS = 128

# Keypoints
NUM_KEYPOINTS = 3
NUM_CLASSES = 4

# Class indices (MUST match CLASS_NAMES order!)
SPRAY_IDX = 0  # nozzle_spray (measurable, has kpts)
NO_SPRAY_IDX = 1  # nozzle_no_spray
FACING_IDX = 2  # nozzle_facing (spraying but non-measurable)
BIG_IDX = 3  # big_nozzle

# Inference thresholds
OBJ_THRESH = 0.35
NMS_IOU_THRESH = 0.5
OBS_THRESH = 0.3


# ============================================================================
# LOSS WEIGHTS (PATCHED: kpt 10 -> 30 for cell-aware encoding)
# ============================================================================
LOSS_WEIGHTS = {
    "obj": 1.0,  # centerness BCE
    "box": 8.0,  # GIoU
    "cls": 2.0,  # 4-class CE (Quality Focal Loss)
    "kpt": 30.0,  # smooth_l1 on cell-relative offsets (raised from 10)
    "obs": 2.0,  # observability BCE
    "iou": 2.0,  # IoU quality BCE
}


# ============================================================================
# CLASS NAMES & EXCLUDED CATS
# ============================================================================
CLASS_NAMES = ["nozzle_spray", "nozzle_no_spray", "nozzle_facing", "big_nozzle"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

# v9: drop nozzle_buildup from core training
EXCLUDED_CATS = {"nozzle_buildup"}

# Spraying classes (for observability head masking)
SPRAYING_CLASSES = {SPRAY_IDX, FACING_IDX}

# Class weights (spray is rare → upweight)
CLS_WEIGHTS = [1.5, 1.0, 1.0, 1.0]

# Files to exclude (bad annotations)
EXCL_FILES = {
    "20250509_d_annotations.coco.json",
    "20250509_k_annotations.coco.json",
}


# ============================================================================
# DEVICE
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# SUCCESS CRITERIA (thesis targets)
# ============================================================================
TARGETS = {
    "mAP_50": 0.85,
    "cls_f1": 0.90,
    "angle_mae_deg": 5.0,
    "observability_f1": 0.80,
}


def print_config():
    """Print key config for verification."""
    print("=" * 60)
    print(f"AFGNet v9 Configuration (hierarchical conditional masking)")
    print("=" * 60)
    print(f"REPO_ROOT:    {REPO_ROOT}")
    print(f"TRAIN_JSON:   {TRAIN_JSON}")
    print(f"VAL_JSON:     {VAL_JSON}")
    print(f"TEST_JSON:    {TEST_JSON}")
    print(f"TRAIN exists: {TRAIN_JSON.exists()}")
    print(f"VAL exists:   {VAL_JSON.exists()}")
    print(f"RUN_DIR:      {RUN_DIR}")
    print(f"DEVICE:       {DEVICE}")
    print(f"IMG_SIZE:     {IMG_SIZE}")
    print(f"BATCH_SIZE:   {BATCH_SIZE}")
    print(f"TOTAL_EPOCHS: {TOTAL_EPOCHS}")
    print(f"LR:           {LR}")
    print(f"Classes:      {CLASS_NAMES}")
    print(f"Excluded:     {EXCLUDED_CATS}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
