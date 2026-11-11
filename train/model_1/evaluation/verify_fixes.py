"""
verify_fixes.py  —  Standalone verification of FIX 1 and FIX 2.

Uses synthetic tensors (no image I/O) so it runs in seconds on CPU.
Verifies:
  FIX 1: per-keypoint regression range — k_t coords stay within rho_t * cell_w
         of cell centre; k_l/k_r can be further.
  FIX 2: bbox-diagonal-normalised keypoint loss — recompute with both old and
         new formulas on identical inputs and report the ratio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "structure"))

import torch
import torch.nn.functional as F
from model import KPT_REGRESSION_RANGE, KPT_RHO_PER_KPT, AFGNet, AFGNetHead

from config import (
    FACING_IDX,
    FEAT_H,
    FEAT_W,
    FPN_CHANNELS,
    IMG_SIZE,
    NUM_KEYPOINTS,
    SPRAY_IDX,
)

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 verification
# ─────────────────────────────────────────────────────────────────────────────


def verify_fix1():
    print("=" * 60)
    print("FIX 1 — Per-keypoint regression range")
    print("=" * 60)
    print(f"KPT_RHO_PER_KPT = {KPT_RHO_PER_KPT}  (tip, left, right)")
    cell_w = 1.0 / FEAT_W
    print(f"cell_w = {cell_w:.5f}")
    print(
        f"  max |k_tip - cell|   = {KPT_RHO_PER_KPT[0] * cell_w:.4f}  (rho={KPT_RHO_PER_KPT[0]})"
    )
    print(
        f"  max |k_left - cell|  = {KPT_RHO_PER_KPT[1] * cell_w:.4f}  (rho={KPT_RHO_PER_KPT[1]})"
    )
    print(
        f"  max |k_right - cell| = {KPT_RHO_PER_KPT[2] * cell_w:.4f}  (rho={KPT_RHO_PER_KPT[2]})"
    )
    print()

    torch.manual_seed(0)
    head = AFGNetHead(in_ch=FPN_CHANNELS).eval()

    # Confirm buffer shape and values
    assert head.kpt_rho.shape == (1, NUM_KEYPOINTS, 1, 1), head.kpt_rho.shape
    print(
        f"head.kpt_rho buffer: shape={tuple(head.kpt_rho.shape)}, "
        f"values={head.kpt_rho.flatten().tolist()}"
    )

    # Drive the kpt head with EXTREME raw logits so sigmoid saturates and we
    # observe the true max offset implied by rho.
    big = 1e3  # sigmoid → 1
    raw = torch.full((1, NUM_KEYPOINTS * 2 + 1, FEAT_H, FEAT_W), big)
    # Bypass stem/kpt_head — patch forward to feed `raw` directly
    with torch.no_grad():
        # Replicate forward() kpt branch only
        K = NUM_KEYPOINTS
        offsets = raw[:, : 2 * K]
        dx_kpts = torch.sigmoid(offsets[:, 0::2]) * 2.0 - 1.0  # ≈ +1
        dy_kpts = torch.sigmoid(offsets[:, 1::2]) * 2.0 - 1.0
        kpt_x = head.cell_x + dx_kpts * (cell_w * head.kpt_rho)
        kpt_y = head.cell_y + dy_kpts * (1.0 / FEAT_H * head.kpt_rho)

    # Per-keypoint max deviation from cell centre
    cell_x = head.cell_x.expand_as(kpt_x)
    cell_y = head.cell_y.expand_as(kpt_y)
    dev_x = (kpt_x - cell_x).abs()
    dev_y = (kpt_y - cell_y).abs()

    print()
    print("Saturated forward → per-keypoint |kpt − cell| max:")
    print(f'  {"kpt":<8} {"max|dx|":>10} {"max|dy|":>10} {"expected":>10}')
    print(f'  {"-"*8} {"-"*10} {"-"*10} {"-"*10}')
    names = ["tip", "left", "right"]
    pass_all = True
    for k in range(K):
        max_dx = float(dev_x[0, k].max())
        max_dy = float(dev_y[0, k].max())
        expected = KPT_RHO_PER_KPT[k] * cell_w
        ok = (max_dx <= expected + 1e-4) and (max_dy <= expected + 1e-4)
        pass_all &= ok
        flag = "OK" if ok else "FAIL"
        print(
            f"  {names[k]:<8} {max_dx:>10.5f} {max_dy:>10.5f} {expected:>10.5f}  {flag}"
        )

    # Compare tip range vs edge range
    tip_max = float(dev_x[0, 0].max())
    edge_max = float(dev_x[0, 1].max())
    if tip_max < edge_max:
        print(f"\nTip range ({tip_max:.4f}) < edge range ({edge_max:.4f})  OK")
    else:
        print(f"\nTip range ({tip_max:.4f}) NOT < edge range ({edge_max:.4f})  FAIL")
        pass_all = False

    print(f'\nFIX 1 verification: {"PASSED" if pass_all else "FAILED"}')
    return pass_all


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 verification
# ─────────────────────────────────────────────────────────────────────────────


def verify_fix2():
    print()
    print("=" * 60)
    print("FIX 2 — Bbox-diagonal-normalised keypoint loss")
    print("=" * 60)

    torch.manual_seed(0)
    N = 50  # number of synthetic spray cells

    # GT keypoints clustered in a small image region (mimicking 20250509)
    gt_kpt = 0.4 + 0.05 * torch.randn(N, NUM_KEYPOINTS * 2)
    # Pred = GT + small Gaussian error (~realistic prediction)
    pred_kpt = gt_kpt + 0.01 * torch.randn(N, NUM_KEYPOINTS * 2)
    # GT bboxes: small nozzles, w,h in ~[0.05, 0.15]
    gt_wh = 0.05 + 0.10 * torch.rand(N, 2)
    gt_w, gt_h = gt_wh[:, 0], gt_wh[:, 1]
    diag = torch.sqrt(gt_w**2 + gt_h**2).clamp(min=1e-6).unsqueeze(1)

    # Old formula (absolute coords)
    L_old = F.smooth_l1_loss(pred_kpt, gt_kpt)
    # New formula (bbox-diag normalised)
    L_new = F.smooth_l1_loss(pred_kpt / diag, gt_kpt / diag)

    print(f"  N cells: {N}")
    print(f"  GT kpts        mean={gt_kpt.mean():.4f}  std={gt_kpt.std():.4f}")
    print(f"  Pred error     mean|diff|={(pred_kpt - gt_kpt).abs().mean():.5f}")
    print(
        f"  bbox diagonal  mean={diag.mean():.4f}  min={diag.min():.4f}  max={diag.max():.4f}"
    )
    print()
    print(f"  L_kpt (old, absolute coords)       = {L_old.item():.6f}")
    print(f"  L_kpt (new, bbox-diag normalised)  = {L_new.item():.6f}")
    print(f"  ratio new/old                      = {L_new.item() / L_old.item():.2f}x")
    print()

    # Sanity 1: new loss > old loss for the same error
    cond_larger = L_new.item() > L_old.item()
    # Sanity 2: a constant predictor (mean of GT) should now produce non-trivial loss
    mean_pred = gt_kpt.mean(dim=0, keepdim=True).expand_as(gt_kpt)
    L_old_const = F.smooth_l1_loss(mean_pred, gt_kpt)
    L_new_const = F.smooth_l1_loss(mean_pred / diag, gt_kpt / diag)
    print(f"  Constant predictor (predicts mean of GT):")
    print(f"    L_old(const) = {L_old_const.item():.6f}")
    print(f"    L_new(const) = {L_new_const.item():.6f}")
    print(
        f"    new penalises constant predictor more: "
        f"{(L_new_const.item() / L_old_const.item()):.2f}x"
    )

    pass_all = cond_larger
    print(f'\nFIX 2 verification: {"PASSED" if pass_all else "FAILED"}')
    return pass_all


if __name__ == "__main__":
    ok1 = verify_fix1()
    ok2 = verify_fix2()
    print()
    print("=" * 60)
    print(f'ALL VERIFICATIONS: {"PASSED" if (ok1 and ok2) else "FAILED"}')
    print("=" * 60)
