"""
model.py - AFGNet v9 with CELL-AWARE BBOX + CELL-AWARE KEYPOINTS

PATCH (May 11): keypoint head now mirrors bbox cell-aware encoding.
Forces model to learn visual features instead of position shortcut.

KEYPOINT encoding (per cell at row=i, col=j):
  cell_cx = (j + 0.5) / FEAT_W
  cell_cy = (i + 0.5) / FEAT_H

  For each of 3 keypoints k in {tip, left, right}:
    dx_k = sigmoid(out) * 2 - 1            # offset [-1, +1]
    dy_k = sigmoid(out) * 2 - 1
    pred_kx = cell_cx + dx_k * (cell_w * KPT_REGRESSION_RANGE)
    pred_ky = cell_cy + dy_k * (cell_h * KPT_REGRESSION_RANGE)

KPT_REGRESSION_RANGE: how many cells away can a keypoint be from its cell.
  Bbox uses 2.0 (center of object). Keypoints (tip + edges of spray cone)
  can be further from cell center, so we set this larger.
  At FEAT=40, cell_width=0.025. RANGE=6 → max offset = 0.15 (15% of image),
  which comfortably covers a typical spray cone radius.

Heads:
  - det_head:   centerness(1) + bbox(4) = 5 channels
  - iou_head:   IoU quality
  - cls_head:   4 classes
  - kpt_head:   3 keypoints * 2 + observability(1)
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FEAT_H, FEAT_W, FPN_CHANNELS, IMG_SIZE, NUM_CLASSES, NUM_KEYPOINTS

REGRESSION_RANGE = 2.0  # bbox center constraint
KPT_REGRESSION_RANGE = (
    20  # (legacy) global keypoint constraint, kept for backwards compat
)
# Per-keypoint cell-aware regression range (in cells).  Tip must stay near the
# nozzle body (rho=3 ≈ 7.5% of image); edges of the spray cone can be much
# further from the detection cell (rho=20 ≈ 50% of image).
# Order matches keypoint layout: [tip, left_edge, right_edge]
KPT_RHO_PER_KPT = (3.0, 20.0, 20.0)


def get_cell_grid(H, W, device):
    """Return cell-center grids of shape [H, W] in normalized coords [0, 1]."""
    cell_y = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    cell_x = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    grid_y, grid_x = torch.meshgrid(cell_y, cell_x, indexing="ij")
    return grid_x, grid_y


class MobileNetV4Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.net = timm.create_model(
            "mobilenetv4_conv_small",
            pretrained=pretrained,
            features_only=True,
            out_indices=(3, 4),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            feats = self.net(dummy)
            self.out_channels = [f.shape[1] for f in feats]

    def forward(self, x):
        return self.net(x)


class LightFPN(nn.Module):
    def __init__(self, in_channels, out_channels=FPN_CHANNELS):
        super().__init__()
        c3, c4 = in_channels
        self.lat_p3 = nn.Sequential(
            nn.Conv2d(c3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.lat_p4 = nn.Sequential(
            nn.Conv2d(c4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        p3, p4 = features
        p4_up = F.interpolate(self.lat_p4(p4), size=p3.shape[-2:], mode="nearest")
        return self.merge(self.lat_p3(p3) + p4_up)


class AFGNetHead(nn.Module):
    """Multi-task head with CELL-AWARE bbox AND keypoints."""

    def __init__(self, in_ch=FPN_CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.det_head = nn.Conv2d(128, 5, 1)
        self.iou_head = nn.Conv2d(128, 1, 1)
        self.cls_head = nn.Conv2d(128, NUM_CLASSES, 1)
        # kpt_head outputs (NUM_KEYPOINTS * 2 + 1) channels:
        #   2*K channels for keypoint dx/dy offsets
        #   1 channel for observability
        self.kpt_head = nn.Conv2d(128, NUM_KEYPOINTS * 2 + 1, 1)

        # Pre-compute cell grid (registered as buffer so moves with .to(device))
        grid_x, grid_y = get_cell_grid(FEAT_H, FEAT_W, torch.device("cpu"))
        self.register_buffer("cell_x", grid_x.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]
        self.register_buffer("cell_y", grid_y.unsqueeze(0).unsqueeze(0))

        # Per-keypoint regression range as a fixed buffer of shape [1, K, 1, 1]
        # so broadcasting against [B, K, H, W] offsets is automatic.
        assert (
            len(KPT_RHO_PER_KPT) == NUM_KEYPOINTS
        ), f"KPT_RHO_PER_KPT length {len(KPT_RHO_PER_KPT)} != NUM_KEYPOINTS {NUM_KEYPOINTS}"
        rho = torch.tensor(KPT_RHO_PER_KPT, dtype=torch.float32).view(
            1, NUM_KEYPOINTS, 1, 1
        )
        self.register_buffer("kpt_rho", rho)

    def forward(self, x):
        f = self.stem(x)
        det = self.det_head(f)  # [B, 5, H, W]
        cls = self.cls_head(f)
        kpt_raw = self.kpt_head(f)  # [B, 2K+1, H, W]

        centerness = torch.sigmoid(det[:, 0:1])

        # === CELL-AWARE BBOX (unchanged from previous v9) ===
        dx = torch.sigmoid(det[:, 1:2]) * 2.0 - 1.0  # [-1, +1]
        dy = torch.sigmoid(det[:, 2:3]) * 2.0 - 1.0

        cell_w = 1.0 / FEAT_W
        cell_h = 1.0 / FEAT_H
        pred_cx = self.cell_x + dx * (cell_w * REGRESSION_RANGE)
        pred_cy = self.cell_y + dy * (cell_h * REGRESSION_RANGE)
        pred_cx = pred_cx.clamp(0.0, 1.0)
        pred_cy = pred_cy.clamp(0.0, 1.0)

        pred_w = torch.sigmoid(det[:, 3:4])  # absolute size
        pred_h = torch.sigmoid(det[:, 4:5])

        bbox = torch.cat([pred_cx, pred_cy, pred_w, pred_h], dim=1)

        # === CELL-AWARE KEYPOINTS (NEW) ===
        # kpt_raw layout: [k1_dx, k1_dy, k2_dx, k2_dy, k3_dx, k3_dy, obs]
        K = NUM_KEYPOINTS  # 3
        offsets = kpt_raw[:, : 2 * K]  # [B, 2K, H, W]
        dx_kpts = torch.sigmoid(offsets[:, 0::2]) * 2.0 - 1.0  # [B, K, H, W]
        dy_kpts = torch.sigmoid(offsets[:, 1::2]) * 2.0 - 1.0  # [B, K, H, W]

        # FIX 1: per-keypoint range. kpt_rho broadcasts [1,K,1,1] over [B,K,H,W].
        kpt_x = self.cell_x + dx_kpts * (cell_w * self.kpt_rho)
        kpt_y = self.cell_y + dy_kpts * (cell_h * self.kpt_rho)
        kpt_x = kpt_x.clamp(0.0, 1.0)
        kpt_y = kpt_y.clamp(0.0, 1.0)

        # Interleave back to [k1x, k1y, k2x, k2y, k3x, k3y] order
        keypoints = torch.stack([kpt_x, kpt_y], dim=2)  # [B, K, 2, H, W]
        keypoints = keypoints.flatten(1, 2)  # [B, 2K, H, W]

        return {
            "centerness": centerness,
            "bbox": bbox,
            "class_logits": cls.clamp(-100, 100),
            "keypoints": keypoints,
            "observability": torch.sigmoid(kpt_raw[:, -1:]),
            "iou_quality": torch.sigmoid(self.iou_head(f)),
        }


class AFGNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = MobileNetV4Backbone(pretrained=pretrained)
        self.fpn = LightFPN(self.backbone.out_channels, FPN_CHANNELS)
        self.head = AFGNetHead(FPN_CHANNELS)

    def forward(self, x):
        return self.head(self.fpn(self.backbone(x)))


def count_parameters(model):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = AFGNet(pretrained=False)
    model.eval()
    print(f"Backbone P3/P4 channels: {model.backbone.out_channels}")
    print(f"BBOX REGRESSION_RANGE:     {REGRESSION_RANGE} cells")
    print(f"KPT  REGRESSION_RANGE:     {KPT_REGRESSION_RANGE} cells")
    print(f"Cell width: {1.0/FEAT_W:.4f}")
    print(f"Max bbox offset from cell: {REGRESSION_RANGE/FEAT_W:.4f}")
    print(f"Max kpt  offset from cell: {KPT_REGRESSION_RANGE/FEAT_W:.4f}")

    with torch.no_grad():
        x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        out = model(x)

    print("\nOutput shapes:")
    for k, v in out.items():
        print(f"  {k:<20}: {tuple(v.shape)}")

    # Verify bbox spatial constraint
    bbox = out["bbox"][0]
    cell_x = model.head.cell_x[0, 0]
    cell_y = model.head.cell_y[0, 0]
    diff_x = (bbox[0] - cell_x).abs()
    diff_y = (bbox[1] - cell_y).abs()
    print(f"\nBBOX spatial constraint check:")
    print(f"  |pred_cx - cell_x|: max={diff_x.max():.4f}, mean={diff_x.mean():.4f}")
    print(f"  Theoretical max:    {REGRESSION_RANGE/FEAT_W:.4f}")
    bbox_in_range = (diff_x <= REGRESSION_RANGE / FEAT_W + 1e-6).all() and (
        diff_y <= REGRESSION_RANGE / FEAT_H + 1e-6
    ).all()
    print(
        f'  All within range:   {bool(bbox_in_range)} {"OK" if bbox_in_range else "BUG"}'
    )

    # Verify kpt spatial constraint
    kpts = out["keypoints"][0]  # [2K, H, W]
    tip_x = kpts[0]
    tip_y = kpts[1]
    kpt_dx = (tip_x - cell_x).abs()
    kpt_dy = (tip_y - cell_y).abs()
    print(f"\nKPT spatial constraint check (tip):")
    print(f"  |pred_tipx - cell_x|: max={kpt_dx.max():.4f}, mean={kpt_dx.mean():.4f}")
    print(f"  Theoretical max:      {KPT_REGRESSION_RANGE/FEAT_W:.4f}")
    kpt_in_range = (kpt_dx <= KPT_REGRESSION_RANGE / FEAT_W + 1e-6).all() and (
        kpt_dy <= KPT_REGRESSION_RANGE / FEAT_H + 1e-6
    ).all()
    print(
        f'  All within range:     {bool(kpt_in_range)} {"OK" if kpt_in_range else "BUG"}'
    )

    total, trainable = count_parameters(model)
    print(f"\nTotal:     {total:,}")
    print(f"Trainable: {trainable:,}")
