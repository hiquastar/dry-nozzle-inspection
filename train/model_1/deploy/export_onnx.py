"""
export_onnx.py - Export AFGNet v9 trained checkpoint to ONNX for Pi 5 deployment.

Run from train/train_v9_v1/:
    python export_onnx.py
"""

import sys
from pathlib import Path

import torch
import torch.onnx

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "structure"))

from model import AFGNet

from config import DEVICE, IMG_SIZE, REPO_ROOT

RUN_NAME = "v9_fixes_reverse_seed42"
CKPT_PATH = REPO_ROOT / "train" / "model_1" / "checkpoint" / RUN_NAME / "train_v9_v1.pth"
ONNX_PATH = REPO_ROOT / "train" / "model_1" / "checkpoint" / RUN_NAME / "pi5_package" / "afgnet_v9.onnx"


def main():
    if not CKPT_PATH.exists():
        sys.exit(f"ERROR: checkpoint not found: {CKPT_PATH}")

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {CKPT_PATH}...")
    model = AFGNet(pretrained=False)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    for k in ("head.cell_x", "head.cell_y"):
        state.pop(k, None)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'  checkpoint: ep{ckpt["epoch"]}, composite={ckpt["composite"]:.4f}')

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    print(f"Exporting to {ONNX_PATH} ...")
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        input_names=["image"],
        output_names=[
            "centerness",
            "bbox",
            "class_logits",
            "keypoints",
            "observability",
            "iou_quality",
        ],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,  # use legacy TorchScript exporter (no onnxscript dep)
    )
    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"OK  {ONNX_PATH.name}  ({size_mb:.2f} MB)")

    # Sanity check the export
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": dummy.numpy()})
    print(f"\nONNX outputs:")
    for name, arr in zip(
        [
            "centerness",
            "bbox",
            "class_logits",
            "keypoints",
            "observability",
            "iou_quality",
        ],
        out,
    ):
        print(f"  {name:<14}  shape={arr.shape}")


if __name__ == "__main__":
    main()
