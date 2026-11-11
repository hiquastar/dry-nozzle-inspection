"""
heads.py - step 7 / option 3: learned readers of the mist map.

Both heads receive ONLY a neighbourhood of the mist probability map around a
nozzle tip, sampled with grid_sample so the whole thing is one network:

  StateHead   a HEAD_WIN x HEAD_WIN window of the map centred at the tip
              -> logit "this nozzle is spraying"
  AngleHead   a polar profile: HEAD_RAYS rays over axis +- 90 deg, each ray
              the mean of HEAD_RAD_N samples along r in HEAD_RAD (map cells),
              directions taken in the ISOTROPIC pixel plane (x scaled by
              W/H) so bin k <-> a true angle -> apex angle in radians

The map is detached before sampling (see model.py): the heads learn to read
the evidence, they cannot reshape it to smuggle in nozzle identity.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ANGLE_ASPECT, HEAD_CC_SIGMA, HEAD_RAD, HEAD_RAD_N, HEAD_RAYS, HEAD_WIN, IMG_H, IMG_W, MIST_STRIDE

MAP_H, MAP_W = IMG_H // MIST_STRIDE, IMG_W // MIST_STRIDE


def _sample(prob, pts_norm):
    """prob [B,1,h,w]; pts_norm [B,N,K,2] normalised (x/W, y/H) -> [B,N,K]"""
    grid = pts_norm * 2.0 - 1.0  # grid_sample wants [-1, 1]
    B, N, K, _ = grid.shape
    out = F.grid_sample(prob, grid.view(B, N * K, 1, 2), mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.view(B, N, K)


def window_points(tips_norm):
    """tips_norm [B,N,2] -> [B,N,WIN*WIN,2] sample points of a WIN x WIN window (map cells)"""
    o = (torch.arange(HEAD_WIN, device=tips_norm.device, dtype=torch.float32) - HEAD_WIN / 2 + 0.5)
    oy, ox = torch.meshgrid(o, o, indexing="ij")
    off = torch.stack([ox.reshape(-1) / MAP_W, oy.reshape(-1) / MAP_H], dim=-1)  # cells -> normalised
    return tips_norm.unsqueeze(2) + off.view(1, 1, -1, 2)


def polar_points(tips_norm, axis_deg):
    """tips_norm [B,N,2], axis_deg [B,N] -> [B,N,RAYS*RAD_N,2] normalised sample points.
    Ray directions are uniform in the isotropic plane; a pixel step of s along
    direction theta is (s*cos/W_px, s*sin/H_px) in normalised units."""
    dev = tips_norm.device
    thetas = axis_deg.unsqueeze(-1) + torch.linspace(-90, 90, HEAD_RAYS, device=dev).view(1, 1, -1)  # [B,N,RAYS]
    rs = torch.linspace(HEAD_RAD[0], HEAD_RAD[1], HEAD_RAD_N, device=dev) * MIST_STRIDE  # map cells -> px on 640 frame
    t = torch.deg2rad(thetas).unsqueeze(-1)  # [B,N,RAYS,1]
    dx = torch.cos(t) * rs / IMG_W  # isotropic px -> normalised x
    dy = torch.sin(t) * rs / IMG_H
    pts = torch.stack([dx, dy], dim=-1)  # [B,N,RAYS,RAD_N,2]
    return tips_norm.view(*tips_norm.shape[:2], 1, 1, 2) + pts


class StateHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(HEAD_WIN * HEAD_WIN, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))

    def forward(self, prob_detached, tips_norm):
        """-> logits [B,N]"""
        win = _sample(prob_detached, window_points(tips_norm))  # [B,N,WIN*WIN]
        return self.mlp(win).squeeze(-1)


class AngleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, 9, padding=4), nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(nn.Linear(8 * HEAD_RAYS, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def profile(self, prob_detached, tips_norm, axis_deg):
        B, N = tips_norm.shape[:2]
        pts = polar_points(tips_norm, axis_deg).view(B, N, HEAD_RAYS * HEAD_RAD_N, 2)
        return _sample(prob_detached, pts).view(B, N, HEAD_RAYS, HEAD_RAD_N).mean(-1)  # [B,N,RAYS]

    def forward(self, prob_detached, tips_norm, axis_deg):
        """-> apex angle in radians [B,N] (sigmoid * pi keeps it in (0, 180) deg)"""
        prof = self.profile(prob_detached, tips_norm, axis_deg)
        B, N = prof.shape[:2]
        x = self.conv(prof.view(B * N, 1, HEAD_RAYS)).view(B * N, -1)
        return (torch.sigmoid(self.mlp(x)) * math.pi).view(B, N)


# ---------------------------------------------------------------------------
# Step 7b: coordinate-classification readout (SimCC-style) of the same profile.
# Bin k of the HEAD_RAYS bins <-> angle -90 + k * BIN_DEG relative to the axis
# (matches torch.linspace(-90, 90, HEAD_RAYS) in polar_points).
# ---------------------------------------------------------------------------
BIN_DEG = 180.0 / (HEAD_RAYS - 1)


def edge_bins(apex_rad):
    """apex angle [M] (rad) -> [M,2] fractional bin of the (lower, upper) edge.
    The axis is the bisector of the annotated edges, so the edges sit at -+ apex/2."""
    half = torch.rad2deg(apex_rad) / 2.0
    return torch.stack([(90.0 - half) / BIN_DEG, (90.0 + half) / BIN_DEG], dim=-1)


def cc_targets(apex_rad, sigma=HEAD_CC_SIGMA):
    """-> [M,2,HEAD_RAYS] Gaussian-smoothed one-hot over bins, rows sum to 1"""
    centre = edge_bins(apex_rad).unsqueeze(-1)  # [M,2,1]
    k = torch.arange(HEAD_RAYS, device=apex_rad.device, dtype=torch.float32).view(1, 1, -1)
    g = torch.exp(-0.5 * ((k - centre) / sigma) ** 2)
    return g / g.sum(-1, keepdim=True)


class AngleHeadCC(AngleHead):
    """Same conv trunk over the polar profile; output = 2 x HEAD_RAYS logits
    ("edge 1 is at bin k", "edge 2 is at bin k"). forward -> (apex_rad, logits)."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(8 * HEAD_RAYS, 64), nn.ReLU(inplace=True), nn.Linear(64, 2 * HEAD_RAYS))

    def forward(self, prob_detached, tips_norm, axis_deg):
        prof = self.profile(prob_detached, tips_norm, axis_deg)
        B, N = prof.shape[:2]
        x = self.conv(prof.view(B * N, 1, HEAD_RAYS)).view(B * N, -1)
        logits = self.mlp(x).view(B, N, 2, HEAD_RAYS)
        k = logits.argmax(-1).float()  # [B,N,2] hard readout, as in SimCC
        apex_deg = ((k[..., 1] - k[..., 0]) * BIN_DEG).clamp(min=0.0)
        return torch.deg2rad(apex_deg), logits
