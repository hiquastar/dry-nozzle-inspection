"""
loss.py - AFGNet v9 multi-task loss with HIERARCHICAL CONDITIONAL MASKING

Thesis novelty: Tasks are MASKED based on class to formalize partial existence:

| Class           | obj | bbox | cls | kpt   | obs   |
|-----------------|-----|------|-----|-------|-------|
| nozzle_spray    | ✓   | ✓    | ✓   | ✓     | ✓ (1) |
| nozzle_no_spray | ✓   | ✓    | ✓   | -     | -     |
| nozzle_facing   | ✓   | ✓    | ✓   | -     | ✓ (0) |
| big_nozzle      | ✓   | ✓    | ✓   | -     | -     |

Where:
  ✓ (1) = trained with target=1 (measurable)
  ✓ (0) = trained with target=0 (non-measurable)
  -     = masked out (no gradient)

This is the OBSERVABILITY-AWARE multi-task formulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CLS_WEIGHTS,
    FACING_IDX,
    LOSS_WEIGHTS,
    NUM_CLASSES,
    NUM_KEYPOINTS,
    SPRAY_IDX,
)


class GIoULoss(nn.Module):
    """GIoU loss for cxcywh bbox format."""

    def forward(self, pred, target):
        def to_xyxy(b):
            return torch.stack(
                [
                    b[:, 0] - b[:, 2] / 2,
                    b[:, 1] - b[:, 3] / 2,
                    b[:, 0] + b[:, 2] / 2,
                    b[:, 1] + b[:, 3] / 2,
                ],
                dim=1,
            )

        p, t = to_xyxy(pred), to_xyxy(target)
        ix1 = torch.max(p[:, 0], t[:, 0])
        iy1 = torch.max(p[:, 1], t[:, 1])
        ix2 = torch.min(p[:, 2], t[:, 2])
        iy2 = torch.min(p[:, 3], t[:, 3])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
        at = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
        union = ap + at - inter + 1e-7
        iou = inter / union
        ex1 = torch.min(p[:, 0], t[:, 0])
        ey1 = torch.min(p[:, 1], t[:, 1])
        ex2 = torch.max(p[:, 2], t[:, 2])
        ey2 = torch.max(p[:, 3], t[:, 3])
        encl = (ex2 - ex1) * (ey2 - ey1) + 1e-7
        return (1 - (iou - (encl - union) / encl)).mean()


class AFGNetLoss(nn.Module):
    """Multi-task loss with hierarchical masking (v9 novelty)."""

    def __init__(self):
        super().__init__()
        self.giou = GIoULoss()
        self.lw = LOSS_WEIGHTS
        self.cls_weights = torch.tensor(CLS_WEIGHTS).float()

    def forward(self, preds, targets, device):
        B = preds["centerness"].shape[0]

        # ──────────────────────────────────────
        # 1. Objectness (centerness) - all cells
        # ──────────────────────────────────────
        obj_loss = F.binary_cross_entropy(
            preds["centerness"].view(B, -1).float(),
            targets["centerness"].to(device).view(B, -1).float(),
        )

        pos_mask = targets["pos_mask"].to(device).bool().view(-1)
        if pos_mask.sum() == 0:
            return self.lw["obj"] * obj_loss, {
                "obj": obj_loss.item(),
                "box": 0,
                "cls": 0,
                "kpt": 0,
                "obs": 0,
                "iou": 0,
                "spray_cells": 0,
                "spraying_cells": 0,
            }

        def flat(t, ch):
            return t.permute(0, 2, 3, 1).reshape(-1, ch)[pos_mask]

        # ──────────────────────────────────────
        # 2. Box loss (GIoU) - all positive cells
        # ──────────────────────────────────────
        box_loss = self.giou(
            flat(preds["bbox"], 4),
            targets["bbox"].to(device).view(-1, 4)[pos_mask],
        )

        # ──────────────────────────────────────
        # 3. Class loss (Quality Focal Loss) - all positive cells
        # ──────────────────────────────────────
        tgt_cls = targets["class_ids"].to(device).view(-1)[pos_mask]
        _logits = flat(preds["class_logits"], NUM_CLASSES).float()
        _valid = (tgt_cls >= 0) & (tgt_cls < NUM_CLASSES)

        if _valid.sum() > 0 and torch.isfinite(_logits[_valid]).all():
            _quality = (
                targets["centerness"].to(device).view(-1)[pos_mask][_valid].detach()
            )
            _logits_v = _logits[_valid]
            _cls_v = tgt_cls[_valid]
            _one_hot = F.one_hot(_cls_v, NUM_CLASSES).float()
            _soft_tgt = _quality.unsqueeze(1) * _one_hot
            _cls_w = self.cls_weights.to(device)
            _pred_sig = torch.sigmoid(_logits_v)
            _qfl = F.binary_cross_entropy(_pred_sig, _soft_tgt, reduction="none")
            _pt = torch.abs(_soft_tgt - _pred_sig).detach()
            _qfl = (_pt**2) * _qfl
            cls_loss = (_qfl * _cls_w.unsqueeze(0)).mean()
        else:
            cls_loss = torch.tensor(0.0, device=device)

        # ──────────────────────────────────────
        # HIERARCHICAL MASKING (v9 NOVELTY)
        # ──────────────────────────────────────
        spray_mask = tgt_cls == SPRAY_IDX
        facing_mask = tgt_cls == FACING_IDX
        spraying_mask = spray_mask | facing_mask

        # ──────────────────────────────────────
        # 4. Keypoint loss - ONLY on spray cells (measurable)
        # FIX 2: bbox-diagonal normalisation. (pred - gt) is divided by the GT
        # bbox diagonal so error scales relative to nozzle size. This penalises
        # constant predictors (mean-of-train) which previously achieved low loss
        # because all spray keypoints clustered in a narrow image region.
        # ──────────────────────────────────────
        if spray_mask.sum() > 0:
            pred_kpt = flat(preds["keypoints"], NUM_KEYPOINTS * 2)[spray_mask]
            gt_kpt = (
                targets["keypoints"]
                .to(device)
                .view(-1, NUM_KEYPOINTS * 2)[pos_mask][spray_mask]
            )

            gt_bbox = (
                targets["bbox"].to(device).view(-1, 4)[pos_mask][spray_mask]
            )  # [N, 4] in cxcywh
            w = gt_bbox[:, 2]
            h = gt_bbox[:, 3]
            diag = torch.sqrt(w * w + h * h).clamp(min=1e-6).unsqueeze(1)  # [N, 1]

            kpt_loss = F.smooth_l1_loss(pred_kpt / diag, gt_kpt / diag)
        else:
            kpt_loss = torch.tensor(0.0, device=device)

        # ──────────────────────────────────────
        # 5. Observability loss - ONLY on spraying cells (spray + facing)
        #    Target: spray = 1 (measurable), facing = 0 (non-measurable)
        # ──────────────────────────────────────
        if spraying_mask.sum() > 0:
            obs_pred = (
                flat(preds["observability"], 1)[spraying_mask].squeeze(-1).float()
            )
            obs_tgt = (tgt_cls[spraying_mask] == SPRAY_IDX).float()
            obs_loss = F.binary_cross_entropy(obs_pred, obs_tgt)
        else:
            obs_loss = torch.tensor(0.0, device=device)

        # ──────────────────────────────────────
        # 6. IoU quality loss - all positive cells
        # ──────────────────────────────────────
        pred_b = flat(preds["bbox"], 4).detach()
        gt_b = targets["bbox"].to(device).view(-1, 4)[pos_mask]

        def _xyxy(b):
            return torch.stack(
                [
                    b[:, 0] - b[:, 2] / 2,
                    b[:, 1] - b[:, 3] / 2,
                    b[:, 0] + b[:, 2] / 2,
                    b[:, 1] + b[:, 3] / 2,
                ],
                1,
            )

        pb, gb = _xyxy(pred_b), _xyxy(gt_b)
        _inter = (torch.min(pb[:, 2], gb[:, 2]) - torch.max(pb[:, 0], gb[:, 0])).clamp(
            0
        ) * (torch.min(pb[:, 3], gb[:, 3]) - torch.max(pb[:, 1], gb[:, 1])).clamp(0)
        _ap = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
        _ag = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
        iou_tgt = (_inter / (_ap + _ag - _inter + 1e-7)).clamp(0, 1).detach()
        iou_loss = F.binary_cross_entropy(
            flat(preds["iou_quality"], 1).squeeze(1).float(),
            iou_tgt.float(),
        )

        # ──────────────────────────────────────
        # Total
        # ──────────────────────────────────────
        total = (
            self.lw["obj"] * obj_loss
            + self.lw["box"] * box_loss
            + self.lw["cls"] * cls_loss
            + self.lw["kpt"] * kpt_loss
            + self.lw["obs"] * obs_loss
            + self.lw["iou"] * iou_loss
        )

        return total, {
            "obj": obj_loss.item(),
            "box": box_loss.item(),
            "cls": cls_loss.item(),
            "kpt": kpt_loss.item(),
            "obs": obs_loss.item(),
            "iou": iou_loss.item(),
            "spray_cells": int(spray_mask.sum().item()),
            "spraying_cells": int(spraying_mask.sum().item()),
        }


if __name__ == "__main__":
    print("Loss with hierarchical masking - smoke test")

    # Synthetic preds (with grad)
    B, H, W = 2, 40, 40
    preds = {
        "centerness": torch.rand(B, 1, H, W, requires_grad=True),
        "bbox": torch.rand(B, 4, H, W, requires_grad=True),
        "class_logits": torch.randn(B, NUM_CLASSES, H, W, requires_grad=True),
        "keypoints": torch.rand(B, NUM_KEYPOINTS * 2, H, W, requires_grad=True),
        "observability": torch.rand(B, 1, H, W, requires_grad=True),
        "iou_quality": torch.rand(B, 1, H, W, requires_grad=True),
    }

    # Synthetic targets
    targets = {
        "centerness": torch.zeros(B, H, W),
        "pos_mask": torch.zeros(B, H, W),
        "bbox": torch.zeros(B, H, W, 4),
        "class_ids": torch.zeros(B, H, W, dtype=torch.long),
        "keypoints": torch.zeros(B, H, W, NUM_KEYPOINTS * 2),
        "observability": torch.zeros(B, H, W),
    }

    # Plant some positives
    targets["pos_mask"][0, 20, 20] = 1
    targets["centerness"][0, 20, 20] = 0.9
    targets["bbox"][0, 20, 20] = torch.tensor([0.5, 0.5, 0.2, 0.2])
    targets["class_ids"][0, 20, 20] = SPRAY_IDX
    targets["observability"][0, 20, 20] = 1.0

    targets["pos_mask"][0, 25, 25] = 1
    targets["centerness"][0, 25, 25] = 0.9
    targets["bbox"][0, 25, 25] = torch.tensor([0.6, 0.6, 0.15, 0.15])
    targets["class_ids"][0, 25, 25] = FACING_IDX

    loss_fn = AFGNetLoss()
    total, breakdown = loss_fn(preds, targets, torch.device("cpu"))

    print(f"\nTotal loss: {total.item():.4f}")
    print("Breakdown:")
    for k, v in breakdown.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")

    # Verify gradient flows
    total.backward()
    print("\n✓ Backward OK - gradients computed")
