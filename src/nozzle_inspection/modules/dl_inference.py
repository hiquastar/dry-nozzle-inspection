"""
Deep learning inference wrappers: AFGNet v9 (one ONNX file) and AFGNet v10 (the
final thesis model, through the Pi 5 package). Use load_afgnet() to pick one.

Loads the ONNX model once, runs full-frame detection on demand, decodes the
6-head FCOS output to a list of detections with class_id in {0,1,2,3}
mapping to {nozzle_spray, nozzle_no_spray, nozzle_facing, big_nozzle}.

Thread-safe for the 0.5s background ticker.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    ort = None
    _ORT_AVAILABLE = False


IMG_SIZE = 640
NMS_IOU_THRESH = 0.50

NUM_CLASSES = 4
CLASS_NAMES = ["nozzle_spray", "nozzle_no_spray", "nozzle_facing", "big_nozzle"]
SPRAY_IDX, NO_SPRAY_IDX, FACING_IDX, BIG_IDX = 0, 1, 2, 3

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_EXPECTED_OUTPUTS = [
    "centerness",
    "bbox",
    "class_logits",
    "keypoints",
    "observability",
    "iou_quality",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _box_iou(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-7)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores)
    keep: List[int] = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([_box_iou(tuple(boxes[i]), tuple(boxes[j])) for j in rest])
        order = rest[ious < iou_thresh]
    return keep


class AFGNetInference:
    """Thread-safe wrapper around onnxruntime AFGNet v9 session.

    Lazy-loads on first inference. Returns detections in pixel coords of the
    original input frame (not normalized).
    """

    def __init__(
        self,
        onnx_path: str,
        n_threads: int = 4,
        obj_threshold: float = 0.10,
        nms_iou: float = NMS_IOU_THRESH,
    ):
        self.onnx_path = onnx_path
        self.n_threads = int(max(1, n_threads))
        self.obj_threshold = float(obj_threshold)
        self.nms_iou = float(nms_iou)
        self._session: Optional[Any] = None
        self._input_name: Optional[str] = None
        self._output_names: Optional[List[str]] = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        if not _ORT_AVAILABLE:
            return False
        if not os.path.isfile(self.onnx_path):
            return False
        return True

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _ensure_loaded(self) -> bool:
        if self._session is not None:
            return True
        if not _ORT_AVAILABLE:
            self._load_error = "onnxruntime not installed"
            return False
        if not os.path.isfile(self.onnx_path):
            self._load_error = f"model file not found: {self.onnx_path}"
            return False
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.n_threads
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                self.onnx_path,
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
            return True
        except Exception as exc:
            self._load_error = f"failed to load onnx session: {exc}"
            self._session = None
            return False

    @staticmethod
    def _preprocess_bgr(bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = rgb.astype(np.float32) * (1.0 / 255.0)
        img = (img - _MEAN) / _STD
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(img)

    def _decode(
        self, outs: Dict[str, np.ndarray], frame_w: int, frame_h: int
    ) -> List[Dict[str, Any]]:
        centerness = outs["centerness"][0, 0]
        bbox = outs["bbox"][0]
        iou_q = outs["iou_quality"][0, 0]
        cls_logits = outs["class_logits"][0]
        obs = outs["observability"][0, 0]
        keypoints = outs["keypoints"][0]

        obj_score_map = centerness * iou_q
        mask = obj_score_map >= self.obj_threshold
        if not mask.any():
            return []

        iy, ix = np.where(mask)
        candidates: List[Dict[str, Any]] = []
        for r, c in zip(iy, ix):
            cx = float(bbox[0, r, c])
            cy = float(bbox[1, r, c])
            bw = float(bbox[2, r, c])
            bh = float(bbox[3, r, c])
            x1 = max(0.0, cx - bw / 2)
            y1 = max(0.0, cy - bh / 2)
            x2 = min(1.0, cx + bw / 2)
            y2 = min(1.0, cy + bh / 2)
            if x2 <= x1 or y2 <= y1:
                continue
            cls_probs = _sigmoid(cls_logits[:, r, c])
            cls_id = int(np.argmax(cls_probs))
            cls_conf = float(cls_probs[cls_id])
            obj_score = float(obj_score_map[r, c])
            candidates.append(
                {
                    "box_norm": (x1, y1, x2, y2),
                    "score": obj_score * cls_conf,
                    "obj_score": obj_score,
                    "cls_conf": cls_conf,
                    "class_id": cls_id,
                    "class_probs": cls_probs.tolist(),
                    "observability": float(obs[r, c]),
                    "keypoints_norm": keypoints[:, r, c].tolist(),
                }
            )

        if not candidates:
            return []

        boxes = np.array([cnd["box_norm"] for cnd in candidates])
        scores = np.array([cnd["score"] for cnd in candidates])
        keep_idx = _nms(boxes, scores, self.nms_iou)

        results: List[Dict[str, Any]] = []
        for i in keep_idx:
            cnd = candidates[i]
            x1n, y1n, x2n, y2n = cnd["box_norm"]
            kx = cnd["keypoints_norm"]
            results.append(
                {
                    "box": (
                        int(round(x1n * frame_w)),
                        int(round(y1n * frame_h)),
                        int(round(x2n * frame_w)),
                        int(round(y2n * frame_h)),
                    ),
                    "box_norm": cnd["box_norm"],
                    "score": cnd["score"],
                    "obj_score": cnd["obj_score"],
                    "cls_conf": cnd["cls_conf"],
                    "class_id": cnd["class_id"],
                    "class_name": CLASS_NAMES[cnd["class_id"]],
                    "class_probs": cnd["class_probs"],
                    "observability": cnd["observability"],
                    "keypoints": [
                        int(round(kx[0] * frame_w)),
                        int(round(kx[1] * frame_h)),
                        int(round(kx[2] * frame_w)),
                        int(round(kx[3] * frame_h)),
                        int(round(kx[4] * frame_w)),
                        int(round(kx[5] * frame_h)),
                    ],
                }
            )
        return results

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run full-frame detection on a BGR image.

        Returns list of detections in pixel coords of the input frame.
        Returns [] on any load/runtime error.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        with self._lock:
            if not self._ensure_loaded():
                return []
            h, w = frame_bgr.shape[:2]
            try:
                x = self._preprocess_bgr(frame_bgr)
                outs = self._session.run(None, {self._input_name: x})
                out_map = dict(zip(self._output_names, outs))
                return self._decode(out_map, w, h)
            except Exception as exc:
                self._load_error = f"inference failure: {exc}"
                return []


class AFGNetV10Inference:
    """AFGNet v10 (final thesis model) behind the same interface as AFGNetInference.

    Runs `FinalSystem` from the Pi 5 package (deployment/pi5_final/pipeline.py), so the
    live app uses the exact decode, mist gate and angle reader of Chapter 6.
    `class_id` is the final state after the mist gate; spray lances also get
    `angle_deg` from the angle reader.
    """

    def __init__(self, package_dir: str, seed: int = 42, n_threads: int = 4):
        self.onnx_path = package_dir  # kept for messages; this is the package folder
        self.package_dir = package_dir
        self.seed = int(seed)
        self.n_threads = int(max(1, n_threads))
        self._system: Optional[Any] = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        if not _ORT_AVAILABLE:
            return False
        cfg = os.path.join(self.package_dir, "config.json")
        if not os.path.isfile(cfg) or not os.path.isfile(
            os.path.join(self.package_dir, "pipeline.py")
        ):
            return False
        import json

        seeds = json.load(open(cfg, encoding="utf-8")).get("seeds", {})
        s = seeds.get(str(self.seed))
        return bool(s) and all(
            os.path.isfile(os.path.join(self.package_dir, "models", s[k]))
            for k in ("main", "reader")
        )

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _ensure_loaded(self) -> bool:
        if self._system is not None:
            return True
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "afgnet_v10_pipeline", os.path.join(self.package_dir, "pipeline.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._system = module.FinalSystem(
                self.package_dir, seed=self.seed, threads=self.n_threads
            )
            return True
        except Exception as exc:
            self._load_error = f"failed to load AFGNet v10: {exc}"
            self._system = None
            return False

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Full-frame detection in pixel coords of the input frame; [] on any error."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        with self._lock:
            if not self._ensure_loaded():
                return []
            h, w = frame_bgr.shape[:2]
            try:
                dets, _ = self._system(frame_bgr)
            except Exception as exc:
                self._load_error = f"inference failure: {exc}"
                return []
        results: List[Dict[str, Any]] = []
        for d in dets:
            x1n, y1n, x2n, y2n = d["box"]
            kx = d["keypoints"]
            state = int(d["state"])
            out = {
                "box": (
                    int(round(x1n * w)),
                    int(round(y1n * h)),
                    int(round(x2n * w)),
                    int(round(y2n * h)),
                ),
                "box_norm": tuple(d["box"]),
                "score": float(d["score"]),
                "obj_score": float(d["obj_score"]),
                "class_id": state,
                "class_name": CLASS_NAMES[state],
                "head_class_id": int(d["class_id"]),
                "observability": float(d["observability"]),
                "keypoints": [
                    int(round(kx[i] * (w if i % 2 == 0 else h))) for i in range(6)
                ],
            }
            if "gate" in d:
                out["mist_gate"] = float(d["gate"])
            if "angle_deg" in d:
                out["angle_deg"] = float(d["angle_deg"])
            results.append(out)
        return results


def load_afgnet(
    path: str, n_threads: int = 4, obj_threshold: float = 0.10, seed: int = 42
):
    """A folder (deployment/pi5_final) loads AFGNet v10; an .onnx file loads AFGNet v9."""
    if os.path.isdir(path):
        return AFGNetV10Inference(path, seed=seed, n_threads=n_threads)
    return AFGNetInference(path, n_threads=n_threads, obj_threshold=obj_threshold)


def map_detections_to_rois(
    detections: List[Dict[str, Any]],
    rois: List[Tuple[int, int, int, int]],
    iou_min: float = 0.10,
    containment_min: float = 0.50,
) -> Dict[int, Dict[str, Any]]:
    """Associate detections to existing ROIs by box overlap.

    Each ROI is matched with the detection that has highest IoU OR has its
    center contained inside the ROI. Returns dict keyed by ROI index ->
    detection dict (or absent if no match).

    rois: list of (x, y, w, h) integer rectangles.
    """
    result: Dict[int, Dict[str, Any]] = {}
    if not detections or not rois:
        return result
    roi_boxes = [(rx, ry, rx + rw, ry + rh) for (rx, ry, rw, rh) in rois]
    used = set()
    for roi_idx, rb in enumerate(roi_boxes):
        best_score = -1.0
        best_det = None
        best_di = -1
        for di, det in enumerate(detections):
            if di in used:
                continue
            db = det["box"]
            iou = _box_iou(db, rb)
            dcx = 0.5 * (db[0] + db[2])
            dcy = 0.5 * (db[1] + db[3])
            contained = (rb[0] <= dcx <= rb[2]) and (rb[1] <= dcy <= rb[3])
            score = iou
            if contained:
                score = max(score, containment_min)
            if score > best_score and (iou >= iou_min or contained):
                best_score = score
                best_det = det
                best_di = di
        if best_det is not None:
            result[roi_idx] = best_det
            used.add(best_di)
    return result


def gate_from_class_id(class_id: int) -> Dict[str, bool]:
    """Single source of truth for the 4-class -> measurement gate.

    Returns: {'measure_buildup': bool, 'measure_spray': bool, 'skip_all': bool}
    """
    if class_id == BIG_IDX:
        return {"measure_buildup": False, "measure_spray": False, "skip_all": True}
    if class_id == SPRAY_IDX:
        return {"measure_buildup": True, "measure_spray": True, "skip_all": False}
    # NO_SPRAY (1) or FACING (2)
    return {"measure_buildup": True, "measure_spray": False, "skip_all": False}
