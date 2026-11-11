"""
pack_pi5.py - complete deployment/pi5_final for the Raspberry Pi 5 after
export_onnx_final.py:

  data/video_cam3_v2.mp4      the operating video of the Chapter 5 protocol:
                              data/videos/20250509_5fps.mp4 (CAM3 V2, a test
                              recording, 2048x1536, 5 fps), copied unchanged; the
                              same video as the earlier Pi benchmark of AFGNet v9
  reference/video_s{seed}.json  outputs of the package on this PC for the first
                              warm-up + timed frames of that video, which
                              benchmark.py compares with the outputs on the Pi

    python deploy/pack_pi5.py            (from train/model_2)
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "deployment" / "pi5_final"
sys.path.insert(0, str(PKG))
from benchmark import summary_of  # noqa: E402
from pipeline import FinalSystem  # noqa: E402

VIDEO = REPO / "data" / "videos" / "20250509_5fps.mp4"
N_FRAMES = 120  # 20 warm-up + 100 timed


def main():
    (PKG / "data").mkdir(exist_ok=True)
    dst = PKG / "data" / "video_cam3_v2.mp4"
    shutil.copyfile(VIDEO, dst)
    (PKG / "reference").mkdir(exist_ok=True)
    for seed in (42, 123):
        system = FinalSystem(PKG, seed=seed)
        cap = cv2.VideoCapture(str(dst))
        outputs = []
        for _ in range(N_FRAMES):
            ok, frame = cap.read()
            assert ok
            outputs.append(summary_of(system(frame)[0]))
        ref = {"seed": seed, "video": dst.name, "frames": N_FRAMES, "onnxruntime": ort.__version__,
               "opencv": cv2.__version__, "outputs": outputs}
        json.dump(ref, open(PKG / "reference" / f"video_s{seed}.json", "w"))
        n = [len(o) for o in outputs]
        states = [d["state"] for o in outputs for d in o]
        print(f"s{seed}: {sum(n)} detections in {N_FRAMES} frames, states {[states.count(k) for k in range(4)]} "
              f"(spray, no_spray, facing, big)")


if __name__ == "__main__":
    main()
