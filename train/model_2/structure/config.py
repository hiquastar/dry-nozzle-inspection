"""
config.py - AFGNet v10 configuration (model_2)

Forked from train_v9_v1/config.py after step 0 of docs/notes/PLAN-V10.md.
Paths point inside train/model_2 so model 1 is never touched.

PATCHED (May 11): LOSS_WEIGHTS['kpt'] raised from 10 -> 30 to compensate
for the smaller numeric scale of cell-aware keypoint offsets (offsets are
~0.05 vs absolute coords ~0.5, so smooth_l1 gradients are ~10x smaller).

AFGNet v9 = cxcywh detector + hierarchical conditional masking (thesis novelty).
"""

import os
from datetime import datetime
from pathlib import Path

import torch

# ============================================================================
# PATHS
# ============================================================================
# model_2/structure/config.py -> structure -> model_2 -> train -> repo root.
# Training runs execute from a code snapshot copied into their run directory
# (checkpoint/<run>/code/), so the root is passed explicitly there.
REPO_ROOT = Path(os.environ.get("AFGNET_REPO_ROOT", Path(__file__).resolve().parents[3]))
MODEL2_ROOT = REPO_ROOT / "train" / "model_2"

# Reverse split: CAM2 in train, CAM1 split into val/test.
# model_2 keeps its own copy so it never depends on data/ moving.
SPLITS_ROOT = MODEL2_ROOT / "data"
TRAIN_JSON = SPLITS_ROOT / "train.coco.json"
VAL_JSON = SPLITS_ROOT / "val.coco.json"
TEST_JSON = SPLITS_ROOT / "test.coco.json"

# Image roots (where images live)
ORIGINAL_IMG_ROOT = REPO_ROOT / "data" / "images"
CONVERTED_IMG_ROOT = REPO_ROOT / "data" / "annotations" / "v2"

# Run output. Every v10 run lives under model_2/checkpoint/<name>;
# pass --run_name to train.py for anything other than the reference.
VERSION = "v10_ref"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = MODEL2_ROOT / "checkpoint" / VERSION
TRAIN_VERSION = "afgnet_v10"
MODEL_PATH = RUN_DIR / f"{TRAIN_VERSION}.pth"
ONNX_PATH = RUN_DIR / f"{TRAIN_VERSION}.onnx"
METRICS_LOG = RUN_DIR / "metrics.json"

# Resume from checkpoint (None = train from ImageNet pretrained)
RESUME_FROM = None


# ============================================================================
# v10 SWITCHES  (docs/notes/PLAN-V10.md, step 1)
# ============================================================================
# All four are ON for v10. Setting the environment variable AFGNET_V9_MODE=1
# turns all four OFF: that reproduces the v9 code path and is the gate test
# (two runs, same seed, must be bit-identical) before anything is trusted.
_V9_MODE = os.environ.get("AFGNET_V9_MODE", "0") == "1"


def _flag(env_name, default):
    """Per-switch override for ablations, e.g. AFGNET_INSTANCE_NORM=0."""
    v = os.environ.get(env_name)
    return default if v is None else v == "1"


NEW_RHO = _flag("AFGNET_NEW_RHO", not _V9_MODE)  # C: rho_box 2->10, rho_kpt (3,20,20)->(9,14,14)
LETTERBOX = _flag("AFGNET_LETTERBOX", not _V9_MODE)  # D: 640x480 input (4:3 kept) instead of 640x640 squash
GEO_AUG = _flag("AFGNET_GEO_AUG", not _V9_MODE)  # E: rotation / shift / isotropic scale on train images
INSTANCE_NORM = _flag("AFGNET_INSTANCE_NORM", not _V9_MODE)  # F: every GT instance weighs the same in the loss

# ============================================================================
# IMAGE / TRAINING HYPERPARAMETERS
# ============================================================================
IMG_W = 640
IMG_H = 480 if LETTERBOX else 640
IMG_SIZE = IMG_W  # legacy name, width only
BATCH_SIZE = 16
# Data loading was the bottleneck (GPU 37%, CPU-bound): 8 cores / 16 threads.
# NOTE: num_workers is part of the reproducibility config - the per-worker
# RNG streams change with the worker count, so a run is bit-reproducible
# only with the same value.
NUM_WORKERS = 12

# Image cache: frames pre-resized to IMG_W x IMG_H (cv2 INTER_AREA) by
# structure/build_cache.py. Decoding a 2048x1536 frame and warping it at full
# resolution cost ~10x more CPU than the network needs. Augmentation then
# runs on the small image; labels are coordinates, so they are unaffected.
# Off in V9 mode so the gate path stays bit-identical to the original loader.
USE_IMG_CACHE = not _V9_MODE
IMG_CACHE_DIR = MODEL2_ROOT / "data" / f"cache_{IMG_W}x{IMG_H}"
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

# Feature map (FCOS detection). 40x30 with letterbox, 40x40 without.
FEAT_STRIDE = 16
FEAT_H = IMG_H // FEAT_STRIDE
FEAT_W = IMG_W // FEAT_STRIDE
FPN_CHANNELS = 128

# Cell-aware regression ranges, in cells. Derived from the annotation
# statistics with rho = ceil(max reach / 0.85), see docs/notes/PARAMETER-RATIONALE.md
# and results/hparam_evidence.json. The v9 values cannot
# represent 82.8% of box centres and 64.4% of tips (tanh would need > 1).
REGRESSION_RANGE = 10.0 if NEW_RHO else 2.0
KPT_RHO_PER_KPT = (9.0, 14.0, 14.0) if NEW_RHO else (3.0, 20.0, 20.0)  # tip, left, right

# Geometric augmentation (train only, applied in source pixel space, then
# normalised). Apex angle is invariant to all three, so labels stay exact.
GEO_AUG_ROT_DEG = 15.0
GEO_AUG_SHIFT_FRAC = 0.10
GEO_AUG_SCALE = (0.8, 1.2)  # MUST be isotropic; anisotropic scale changes the angle

# ── Step 3 / 4 switches (docs/notes/PLAN-V10.md), default OFF: each is one
# ablation on top of the clean reference. Override with AFGNET_<NAME>=1.
ANGLE_LOSS = _flag("AFGNET_ANGLE_LOSS", False)  # 3a: penalise the apex angle computed from predicted keypoints
TIP_RELATIVE_EDGES = _flag("AFGNET_TIP_RELATIVE_EDGES", False)  # 3b: decode cone edges relative to the predicted tip
# Range for 3b, in cells, same rule as the other rhos: tip->edge distance in
# the annotations is at most 3.64 cells on the 40x30 grid -> ceil(3.64/0.85).
EDGE_RHO_FROM_TIP = 5.0
TWO_FRAME = _flag("AFGNET_TWO_FRAME", False)  # 4: input = frame t and frame t-FRAME_GAP (6 channels)
# 3c: "angle augmentation". Every nozzle has ONE angle in the data, so
# memorising it is the loss minimum. A random HORIZONTAL stretch on frames
# that contain a spray gives the same nozzle many apex angles; the label is
# not invented - it is recomputed from the stretched keypoints. Consequence:
# the x-stretch counterfactual is no longer an independent probe; the
# pre-registered readouts for 3c are the y-stretch and shear slopes.
ANGLE_AUG = _flag("AFGNET_ANGLE_AUG", False)
ANGLE_AUG_SCALE = (0.8, 1.25)
# 2G: strong photometric augmentation. Measured (plan section 2G): ring
# brightness around the nozzle separates facing/no_spray with AUC 0.98 on the
# training camera but 0.62 on the validation camera, and the dark video
# 20250508 contains only no_spray - absolute brightness is a camera-specific
# shortcut. Random gamma / brightness / contrast / grayscale make absolute
# brightness unreliable during training so the network must use shape.
# Step 6 / option B: MIST head - the network predicts a per-pixel mist map
# (stride 4, 160x120) from pseudo-labels built by build_mist_masks.py.
# State ("spraying?") and cone angle are then DERIVED from the predicted
# map by geometry, not regressed - a map cannot be memorised per nozzle.
MIST_HEAD = _flag("AFGNET_MIST_HEAD", False)
MIST_BG_SIGMA = 40  # px on 640x480: background scale for local-excess brightness
MIST_T = 10.0  # local-excess brightness that counts as mist (cv_state_gate threshold 10.42)
MIST_EDGE_G = 160.0  # Sobel-3 magnitude (~8x per-pixel slope) at which a bright pixel is glare, not mist; 40 also erased soft mist blobs
MIST_STRIDE = 4
MIST_ROI_FACTOR = 0.75  # loss only within this * box size around labelled small nozzles (1.5 covered ~70% of the frame; cone extent is ~0.6 box)
MIST_CACHE_DIR = MODEL2_ROOT / "data" / "cache_640x480_mist"
# Step 7 / option 3: LEARNED readers of the mist map. Two small heads that see
# ONLY a neighbourhood of the (detached) mist map around the nozzle tip:
#   state head  16x16 map window at the tip  -> spraying?   (BCE vs class label)
#   angle head  180-ray polar profile of the map around the tip -> apex angle
# They cannot see the image or the position on the frame, so the only way to
# be right is to read the mist evidence. Compared against the fixed decoders.
LEARNED_HEADS = _flag("AFGNET_LEARNED_HEADS", False)
# Ablation (2026-09-22): where the mist head reads from. Default = the fused
# FPN feature F, before the shared stem (the completed design). 1 = read the
# shared stem output instead, i.e. a fifth head next to the four grid heads;
# the mist loss then also shapes the stem features the other heads use.
MIST_FROM_STEM = _flag("AFGNET_MIST_FROM_STEM", False)
# Ablations 5i (2026-09-24) on the completed design, scored on VAL only.
# NO_MASK: the class-determined masking is switched OFF. The keypoint loss is
# applied on every positive cell, with the placeholder target 0 on cells whose
# class has no keypoints, and the observability loss on every positive cell
# (target 1 on spray, 0 elsewhere). This is the "fill a default value" route
# that Section 4.7 of the thesis argues against.
NO_MASK = _flag("AFGNET_NO_MASK", False)
# OBS_OFF: the observability loss is dropped (weight 0). The head's output
# channel stays in the network, untrained, so every tensor keeps its shape.
OBS_OFF = _flag("AFGNET_OBS_OFF", False)
HEAD_WIN = 16  # map cells (stride 4) -> 64 px window on the 640-wide frame
HEAD_RAYS = 180  # angular bins over axis +- 90 deg
HEAD_RAD = (2.5, 17.5)  # radial range in map cells (10..70 px on the 640 frame)
HEAD_RAD_N = 16
MAX_INST = 12  # per-image instance slots for the head targets
# Readout of the angle head (step 7b): 'reg' = one number (sigmoid*pi, smooth-L1);
# 'cc' = coordinate classification a la SimCC: one HEAD_RAYS-way softmax per
# cone edge over the same polar profile, Gaussian-smoothed one-hot targets
# (sigma in bins), argmax at inference. Same evidence, same labels, only the
# output format changes -> isolates whether the readout is what memorises.
ANGLE_HEAD = os.environ.get("AFGNET_ANGLE_HEAD", "reg")
assert ANGLE_HEAD in ("reg", "cc"), ANGLE_HEAD
HEAD_CC_SIGMA = 2.0
PHOTO_AUG = _flag("AFGNET_PHOTO_AUG", False)
PHOTO_AUG_GAMMA = (0.5, 2.0)
PHOTO_AUG_BRIGHTNESS = 0.5
PHOTO_AUG_CONTRAST = 0.5
PHOTO_AUG_GRAY_P = 0.2
FRAME_GAP = int(os.environ.get("AFGNET_FRAME_GAP", "3"))
IN_CHANS = 6 if TWO_FRAME else 3

# Standard evaluation (also used for checkpoint selection during training)
TIP_MATCH_PX = 50  # angle metric: GT spray <-> prediction by tip distance, source px
ANGLE_MATCH_OBJ_THRESH = 0.05  # decode threshold for the angle metric only

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
    # 3a: apex-angle consistency (radians, smooth-L1). Rule: weight such that
    # its trunk-gradient share equals the kpt term's. Measured on
    # v10_ref_s42 with grad_norm_probe.py (results/grad_norm_v10_ref_s42_angle.json):
    # weight 10 -> angle 28.3% vs kpt 14.5%  =>  10 * 14.5/28.3 = 5.1 -> 5.
    # Only used when ANGLE_LOSS is on.
    "angle": 5.0,
    # step 6 / B: mist-map BCE. Rule: trunk-gradient share equal to the box
    # term. Probe at init (results/grad_norm_init.json, MIST_HEAD on):
    # weight 4 -> mist 3.6% vs box 6.5%  =>  4 * 6.5 / 3.6 = 7.2 -> 7.
    "mist": 7.0,
    # step 7: learned map readers. The map is detached, so these only train
    # the two small heads - no trunk balance needed.
    "state_head": 1.0,
    "angle_head": 1.0,
}
# Ablation 5k (2026-09-25): AFGNET_KPT_WEIGHT overrides LOSS_WEIGHTS['kpt'].
# The 30 was set on 2026-05-11 for the cell-aware encoding at rho_kpt (3,20,20),
# before rho_tip 3->9, tip-relative edges and the bbox-diagonal normalisation
# changed the keypoint gradient scale; 10 is the value it replaced. Default path
# (variable unset) leaves LOSS_WEIGHTS untouched, see PLAN-V10.md §5k.
if os.environ.get("AFGNET_KPT_WEIGHT"):
    LOSS_WEIGHTS["kpt"] = float(os.environ["AFGNET_KPT_WEIGHT"])


# ============================================================================
# CLASS NAMES & EXCLUDED CATS
# ============================================================================
CLASS_NAMES = ["nozzle_spray", "nozzle_no_spray", "nozzle_facing", "big_nozzle"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

# v9: drop nozzle_buildup from core training
EXCLUDED_CATS = {"nozzle_buildup"}

# Spraying classes (for observability head masking)
SPRAYING_CLASSES = {SPRAY_IDX, FACING_IDX}

# Class weights (spray is rare → upweight). The default 1.5 was set by hand.
# Ablation 5j (2026-09-24): AFGNET_CLS_WEIGHTS="w_spray,w_no_spray,w_facing,w_big"
# overrides it with a frequency-derived vector, see PLAN-V10.md §5j.
CLS_WEIGHTS = [1.5, 1.0, 1.0, 1.0]
if os.environ.get("AFGNET_CLS_WEIGHTS"):
    CLS_WEIGHTS = [float(v) for v in os.environ["AFGNET_CLS_WEIGHTS"].split(",")]
    assert len(CLS_WEIGHTS) == 4, CLS_WEIGHTS

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
    print(f"IMG (WxH):    {IMG_W}x{IMG_H}  -> grid {FEAT_W}x{FEAT_H}")
    print(f"v10 switches: NEW_RHO={NEW_RHO} LETTERBOX={LETTERBOX} GEO_AUG={GEO_AUG} INSTANCE_NORM={INSTANCE_NORM}"
          + ("   [AFGNET_V9_MODE]" if _V9_MODE else ""))
    print(f"rho:          box={REGRESSION_RANGE}  kpt={KPT_RHO_PER_KPT}")
    print(f"step 3/4:     ANGLE_LOSS={ANGLE_LOSS} (w={LOSS_WEIGHTS['angle']}) TIP_RELATIVE_EDGES={TIP_RELATIVE_EDGES} "
          f"ANGLE_AUG={ANGLE_AUG} {ANGLE_AUG_SCALE if ANGLE_AUG else ''} PHOTO_AUG={PHOTO_AUG} "
          f"TWO_FRAME={TWO_FRAME} (gap {FRAME_GAP}, in_chans {IN_CHANS})")
    print(f"step 6/7:     MIST_HEAD={MIST_HEAD} LEARNED_HEADS={LEARNED_HEADS} ANGLE_HEAD={ANGLE_HEAD} MIST_FROM_STEM={MIST_FROM_STEM}")
    print(f"abl 5i:       NO_MASK={NO_MASK} OBS_OFF={OBS_OFF}")
    print(f"BATCH_SIZE:   {BATCH_SIZE}")
    print(f"TOTAL_EPOCHS: {TOTAL_EPOCHS}")
    print(f"LR:           {LR}")
    print(f"Classes:      {CLASS_NAMES}")
    print(f"Excluded:     {EXCLUDED_CATS}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
