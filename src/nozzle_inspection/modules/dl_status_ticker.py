"""
Background DL status ticker.

A daemon thread that periodically (every ~0.5s) runs full-frame AFGNet
inference on the latest published frame, maps detections to a fixed set of
ROIs (profile zones), and caches the per-zone class status. The runtime
pipeline reads this cache for gating and overlay.

Frame position never moves - only status updates. This matches "Scenario A"
from the locked design.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .dl_inference import AFGNetInference, map_detections_to_rois


class DLStatusTicker:
    """Background ticker that classifies fixed ROIs every `interval_sec`."""

    def __init__(self, dl: AFGNetInference, interval_sec: float = 0.5):
        self._dl = dl
        self._interval = float(max(0.1, interval_sec))
        self._lock = threading.Lock()
        self._frame_ref = None  # last published frame (BGR)
        self._rois_ref: List[Tuple[int, int, int, int]] = []
        self._zone_id_ref: List[int] = []
        self._status_by_zone: Dict[int, Dict[str, Any]] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="DLStatusTicker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=float(timeout))
        self._thread = None

    # ── Publisher (called from main loop, must be cheap) ──────
    def publish_frame(
        self, frame_bgr, rois: List[Tuple[int, int, int, int]], zone_ids: List[int]
    ) -> None:
        with self._lock:
            self._frame_ref = frame_bgr
            self._rois_ref = list(rois)
            self._zone_id_ref = list(zone_ids)
            # Drop stale status entries for zones that no longer exist
            valid = set(int(z) for z in zone_ids)
            self._status_by_zone = {
                k: v for k, v in self._status_by_zone.items() if int(k) in valid
            }

    # ── Consumer (called from pipeline) ───────────────────────
    def get_status_for_zone(self, zone_id: int) -> Optional[Dict[str, Any]]:
        """Returns the latest detection dict for the zone or None."""
        with self._lock:
            return self._status_by_zone.get(int(zone_id))

    def trigger_now(self) -> int:
        """Run one extra inference now, in the caller's thread. Returns the number of zones classified."""
        return self._tick_once()

    # ── Internal loop ─────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self._interval)
            if self._stop_event.is_set():
                break
            try:
                self._tick_once()
            except Exception as exc:
                # Never let the ticker thread die from a transient error
                print(f"[DLTicker] tick error: {exc}")

    def _tick_once(self) -> int:
        # Snapshot inputs without holding lock during inference
        with self._lock:
            frame = self._frame_ref
            rois = list(self._rois_ref)
            zone_ids = list(self._zone_id_ref)

        if frame is None or not rois:
            return 0

        detections = self._dl.detect(frame)
        mapping = map_detections_to_rois(detections, rois)

        # Build status keyed by zone_id (mapping is keyed by roi index)
        new_status: Dict[int, Dict[str, Any]] = {}
        for roi_idx, det in mapping.items():
            if roi_idx >= len(zone_ids):
                continue
            zid = int(zone_ids[roi_idx])
            new_status[zid] = {
                "class_id": int(det["class_id"]),
                "class_name": det.get("class_name"),
                "score": float(det.get("score", 0.0)),
                "observability": float(det.get("observability", 0.0)),
                "keypoints": det.get("keypoints"),
                "ts": time.time(),
            }

        with self._lock:
            # Merge: keep zones that weren't classified this tick (leave their
            # last-known status in place — stale but not wrong).
            for zid, st in new_status.items():
                self._status_by_zone[zid] = st
        return len(new_status)


def gate_from_status(status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Translate a ticker status entry into measurement gate flags.

    Mirrors `dl_inference.gate_from_class_id` but accepts the cached status
    dict directly. When `status` is None (no DL classification yet), defaults
    to SAFE: measure both buildup and spray — never silently skip.
    """
    if not status or "class_id" not in status:
        return {
            "measure_buildup": True,
            "measure_spray": True,
            "skip_all": False,
            "class_id": None,
            "class_name": None,
            "keypoints": None,
            "reason": "no_dl_status_yet",
        }
    cid = int(status["class_id"])
    from .dl_inference import CLASS_NAMES, gate_from_class_id

    g = gate_from_class_id(cid)
    g["class_id"] = cid
    g["class_name"] = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else None
    g["score"] = float(status.get("score", 0.0))
    g["keypoints"] = status.get("keypoints")
    g["reason"] = "dl_class_" + (g["class_name"] or "unknown")
    return g
