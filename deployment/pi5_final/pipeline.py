"""
pipeline.py - the final system of the thesis (configuration 7) with ONNX Runtime
and numpy only, for the Raspberry Pi 5. No torch is needed.

It reproduces the reported system of Chapter 6 step by step:
  preprocess   2048x1536 BGR frame -> 640x480 (cv2.INTER_AREA, as the evaluation
               cache), RGB, ImageNet normalisation, [1,3,480,640]
  forward      models/afgnet_final_s{seed}.onnx -> decoded grid outputs + mist map
  decode       cells with centerness * iou_quality >= 0.35, greedy NMS at IoU 0.5
               (metrics.decode_predictions of the training code)
  readers      spray and big keep the class of the classification head; every
               other small lance is spraying ("facing") when the mean of the mist
               map in a disk at its tip exceeds the gate threshold of that seed,
               else "no_spray"; on spray lances the angle reader
               (models/angle_reader_s{seed}.onnx) gives the apex angle from the
               mist map at the tip and axis of the keypoints

    from pipeline import FinalSystem
    system = FinalSystem("path/to/pi5_final", seed=42)
    dets, t = system(cv2.imread("frame.png"))    # t: seconds per stage
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HEADS = ("centerness", "bbox", "class_logits", "keypoints", "observability", "iou_quality", "mist_logits")
SPRAY, NO_SPRAY, FACING, BIG = 0, 1, 2, 3


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-7)


def nms(boxes, scores, iou_thresh):
    order, keep = np.argsort(-scores), []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        order = rest[np.array([box_iou(boxes[i], boxes[j]) for j in rest]) < iou_thresh]
    return keep


class FinalSystem:
    def __init__(self, pkg_dir, seed=42, threads=4):
        pkg_dir = Path(pkg_dir)
        self.cfg = cfg = json.load(open(pkg_dir / "config.json"))
        s = cfg["seeds"][str(seed)]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = ["CPUExecutionProvider"]
        self.main = ort.InferenceSession(str(pkg_dir / "models" / s["main"]), so, providers=prov)
        self.reader = ort.InferenceSession(str(pkg_dir / "models" / s["reader"]), so, providers=prov)
        self.seed, self.thr = seed, s["gate_threshold"]
        self.W, self.H = cfg["img_w"], cfg["img_h"]
        self.mean = np.array(cfg["mean"], np.float32)
        self.std = np.array(cfg["std"], np.float32)
        self.yy, self.xx = np.mgrid[0:self.H, 0:self.W]  # pixel grid of the 640x480 map, for the tip disk

    # ---- stages ---------------------------------------------------------------
    def preprocess(self, bgr):
        small = cv2.resize(bgr, (self.W, self.H), interpolation=cv2.INTER_AREA)
        rgb = small[:, :, ::-1].astype(np.float32) / 255.0
        return ((rgb - self.mean) / self.std).transpose(2, 0, 1)[None].copy()

    def forward(self, x):
        return dict(zip(HEADS, self.main.run(None, {"image": x})))

    def decode(self, out):
        c, bbox, iou_q = out["centerness"][0, 0], out["bbox"][0], out["iou_quality"][0, 0]
        cls_logits, kpts, obs = out["class_logits"][0], out["keypoints"][0], out["observability"][0, 0]
        obj = c * iou_q
        cand = []
        for r, cc in zip(*np.where(obj >= self.cfg["obj_thresh"])):
            cx, cy, bw, bh = (float(v) for v in bbox[:, r, cc])
            x1, y1, x2, y2 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2), min(1.0, cx + bw / 2), min(1.0, cy + bh / 2)
            if x2 <= x1 or y2 <= y1:
                continue
            p = 1.0 / (1.0 + np.exp(-cls_logits[:, r, cc]))
            k = int(np.argmax(p))
            cand.append({"box": (x1, y1, x2, y2), "score": float(obj[r, cc]) * float(p[k]), "obj_score": float(obj[r, cc]),
                         "class_id": k, "observability": float(obs[r, cc]), "keypoints": kpts[:, r, cc].tolist()})
        if not cand:
            return []
        keep = nms(np.array([d["box"] for d in cand]), np.array([d["score"] for d in cand]), self.cfg["nms_iou"])
        return [cand[i] for i in keep]

    def tip_disk_mean(self, prob, box):
        W, H = self.W, self.H
        x1, y1, x2, y2 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
        tip = corners[np.argsort(np.linalg.norm(corners - [W / 2, H / 2], axis=1))[:2]].mean(0)
        r = self.cfg["tip_disk"] * max(x2 - x1, y2 - y1)
        # only the window around the disk: the same pixels in the same order as the
        # whole-frame mask of eval_mist_hybrid.tip_disk_mean, so the same mean
        c0, c1 = max(0, int(np.floor(tip[0] - r))), min(W, int(np.ceil(tip[0] + r)) + 1)
        r0, r1 = max(0, int(np.floor(tip[1] - r))), min(H, int(np.ceil(tip[1] + r)) + 1)
        if c1 <= c0 or r1 <= r0:
            return 0.0
        m = np.hypot(self.xx[r0:r1, c0:c1] - tip[0], self.yy[r0:r1, c0:c1] - tip[1]) < r
        return float(prob[r0:r1, c0:c1][m].mean()) if m.sum() > 10 else 0.0

    def readers(self, out, dets):
        mist = out["mist_logits"]
        prob = cv2.resize(1.0 / (1.0 + np.exp(-mist[0, 0])), (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        sprays = []
        for d in dets:
            c = d["class_id"]
            if c == SPRAY:
                d["state"] = SPRAY
                sprays.append(d)
            elif c == BIG:
                d["state"] = BIG
            else:
                d["gate"] = self.tip_disk_mean(prob, d["box"])
                d["state"] = FACING if d["gate"] > self.thr else NO_SPRAY
        if sprays:  # tip and axis from the keypoints, as MH.head_angle; one reader call for all spray lances
            tips, axes = [], []
            for d in sprays:
                k = np.array(d["keypoints"], float).reshape(3, 2)
                iso = (k - k[0]) * [self.cfg["angle_aspect"], 1.0]
                v = iso[1] + iso[2]
                tips.append(k[0]); axes.append(np.degrees(np.arctan2(v[1], v[0])))
            ang = self.reader.run(None, {"mist_logits": mist, "tips": np.array([tips], np.float32),
                                         "axis_deg": np.array([axes], np.float32)})[0][0]
            for d, a in zip(sprays, ang):
                d["angle_deg"] = float(a)
        return dets

    # ---- one frame --------------------------------------------------------------
    def __call__(self, bgr):
        t0 = time.perf_counter()
        x = self.preprocess(bgr)
        t1 = time.perf_counter()
        out = self.forward(x)
        t2 = time.perf_counter()
        dets = self.decode(out)
        t3 = time.perf_counter()
        dets = self.readers(out, dets)
        t4 = time.perf_counter()
        return dets, {"preprocess": t1 - t0, "forward": t2 - t1, "decode_nms": t3 - t2, "readers": t4 - t3}
