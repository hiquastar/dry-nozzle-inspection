"""
export_onnx_final.py - export the final model (configuration 7, abl_heads) to ONNX
for the Raspberry Pi 5 measurement of Chapter 5 / Section 6.4.

The final system runs in two graphs, because the angle reader needs the tip and
axis of each spray lance, which exist only after decoding and NMS:

  afgnet_final_s{seed}.onnx   image [1,3,480,640] (ImageNet-normalised RGB)
                              -> centerness, bbox, class_logits, keypoints,
                                 observability, iou_quality (all decoded as in
                                 model.py), mist_logits [1,1,120,160]
  angle_reader_s{seed}.onnx   mist_logits [1,1,120,160], tips [1,N,2] (normalised),
                              axis_deg [1,N] -> apex angle in degrees [1,N]

Decoding, NMS and the mist gate are plain numpy (deployment/pi5_final/pipeline.py).
Float32, no quantisation, so the ONNX model is the model reported in Chapter 6.

    python deploy/export_onnx_final.py            (from train/model_2; both seeds)
"""

import os

for k, v in {"AFGNET_INSTANCE_NORM": "0", "AFGNET_TIP_RELATIVE_EDGES": "1",
             "AFGNET_MIST_HEAD": "1", "AFGNET_LEARNED_HEADS": "1"}.items():
    os.environ[k] = v  # the flags of the final model, set before config is imported

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "structure"))
sys.path.insert(0, str(HERE.parent / "evaluation"))

import math

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import load_model
from config import (ANGLE_ASPECT, HEAD_RAD, HEAD_RAD_N, HEAD_RAYS, IMG_H, IMG_W, MIST_STRIDE, NMS_IOU_THRESH,
                    OBJ_THRESH, REPO_ROOT)

OUT = REPO_ROOT / "deployment" / "pi5_final" / "models"
RES = REPO_ROOT / "train" / "model_2" / "results"
HEADS = ("centerness", "bbox", "class_logits", "keypoints", "observability", "iou_quality", "mist_logits")


class Main(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return tuple(out[k] for k in HEADS)


class Reader(nn.Module):
    """AngleHead.forward of heads.py with the same weights, written with ops the ONNX
    exporter supports (deg2rad / rad2deg -> multiplication by pi/180); checked against
    the original head in export() below."""

    def __init__(self, head):
        super().__init__()
        self.head = head
        self.register_buffer("offsets", torch.linspace(-90, 90, HEAD_RAYS))
        self.register_buffer("rs", torch.linspace(HEAD_RAD[0], HEAD_RAD[1], HEAD_RAD_N) * MIST_STRIDE)

    def forward(self, mist_logits, tips, axis_deg):
        prob = torch.sigmoid(mist_logits)
        B, N = tips.shape[0], tips.shape[1]
        t = ((axis_deg.unsqueeze(-1) + self.offsets.view(1, 1, -1)) * (math.pi / 180.0)).unsqueeze(-1)  # [B,N,RAYS,1]
        pts = torch.stack([torch.cos(t) * self.rs / IMG_W, torch.sin(t) * self.rs / IMG_H], dim=-1)  # [B,N,RAYS,RAD_N,2]
        pts = tips.view(B, N, 1, 1, 2) + pts
        grid = (pts * 2.0 - 1.0).view(B, N * HEAD_RAYS * HEAD_RAD_N, 1, 2)
        prof = F.grid_sample(prob, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        prof = prof.view(B, N, HEAD_RAYS, HEAD_RAD_N).mean(-1)
        x = self.head.conv(prof.view(B * N, 1, HEAD_RAYS)).view(B * N, -1)
        return (torch.sigmoid(self.head.mlp(x)) * 180.0).view(B, N)  # sigmoid * pi rad, in degrees


def export(seed):
    ckpt = REPO_ROOT / "train" / "model_2" / "checkpoint" / f"abl_heads_s{seed}" / "afgnet_v10.pth"
    model = load_model(ckpt, "cpu").eval()
    OUT.mkdir(parents=True, exist_ok=True)
    main_path, reader_path = OUT / f"afgnet_final_s{seed}.onnx", OUT / f"angle_reader_s{seed}.onnx"

    # the wrappers are put in eval mode: torch.onnx.export restores the wrapper's own mode
    # afterwards, and a wrapper left in train mode would switch BatchNorm of the model to
    # batch statistics for everything that runs after the export
    x = torch.randn(1, 3, IMG_H, IMG_W)
    torch.onnx.export(Main(model).eval(), (x,), str(main_path), input_names=["image"], output_names=list(HEADS),
                      opset_version=17, do_constant_folding=True, dynamo=False)
    mist = torch.randn(1, 1, IMG_H // 4, IMG_W // 4)
    tips, axis = torch.rand(1, 3, 2), torch.rand(1, 3) * 360 - 180
    torch.onnx.export(Reader(model.angle_head).eval(), (mist, tips, axis), str(reader_path),
                      input_names=["mist_logits", "tips", "axis_deg"], output_names=["angle_deg"],
                      dynamic_axes={"tips": {1: "n"}, "axis_deg": {1: "n"}, "angle_deg": {1: "n"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    for p in (main_path, reader_path):
        onnx.checker.check_model(onnx.load(str(p)))

    # raw parity on random inputs, and on the reader with several numbers of lances
    model.eval()
    assert not any(m.training for m in model.modules())
    s = ort.InferenceSession(str(main_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = Main(model).eval()(x)
    got = s.run(None, {"image": x.numpy()})
    main_diff = {k: float(np.abs(r.numpy() - g).max()) for k, r, g in zip(HEADS, ref, got)}
    r = ort.InferenceSession(str(reader_path), providers=["CPUExecutionProvider"])
    reader_diff = {}
    for n in (1, 3, 7):
        t, a = torch.rand(1, n, 2) * 0.8 + 0.1, torch.rand(1, n) * 360 - 180
        with torch.no_grad():  # reference: the original head of the model, as MH.head_angle calls it
            want = np.degrees(model.read_angle(mist, t, a).numpy())
        have = r.run(None, {"mist_logits": mist.numpy(), "tips": t.numpy(), "axis_deg": a.numpy()})[0]
        reader_diff[n] = float(np.abs(want - have).max())
    thr = json.load(open(RES / f"mist_hybrid_test_final_heads_s{seed}.json"))["state"]["gate_threshold_from_train"]
    info = {"seed": seed, "checkpoint": str(ckpt.relative_to(REPO_ROOT)), "main": main_path.name, "reader": reader_path.name,
            "size_mb": {p.name: round(p.stat().st_size / 1e6, 2) for p in (main_path, reader_path)},
            "gate_threshold": thr, "max_abs_diff_main": main_diff, "max_abs_diff_reader_deg": reader_diff}
    print(json.dumps(info, indent=1))
    return info


def main():
    infos = {str(sd): export(sd) for sd in (42, 123)}
    cfg = {  # everything the numpy pipeline needs, taken from config.py of the final model
        "img_w": IMG_W, "img_h": IMG_H, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "resize": "cv2.INTER_AREA (as build_cache.py)", "obj_thresh": OBJ_THRESH, "nms_iou": NMS_IOU_THRESH,
        "tip_disk": 0.45, "angle_aspect": ANGLE_ASPECT,
        "class_names": ["spray", "no_spray", "facing", "big"],
        "seeds": {sd: {"main": i["main"], "reader": i["reader"], "gate_threshold": i["gate_threshold"]} for sd, i in infos.items()},
    }
    json.dump(cfg, open(OUT.parent / "config.json", "w"), indent=1)
    print("wrote", OUT.parent / "config.json")


if __name__ == "__main__":
    main()
