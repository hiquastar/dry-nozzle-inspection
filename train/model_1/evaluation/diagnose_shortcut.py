"""
diagnose_shortcut.py - Verify the position shortcut hypothesis BEFORE refactoring.

Run this on your current trained checkpoint. If outputs confirm shortcut,
proceed with the cell-aware keypoint fix. If not, the bug is elsewhere.

THREE TESTS:
  1. TRANSLATION INVARIANCE: shift the same image, see if keypoints follow.
     Expected if shortcut: predictions stay near original pixel location.
     Expected if features-based: predictions follow the spray.

  2. MASKED BACKGROUND: black out everything except a tight crop around spray.
     Expected if shortcut: predictions collapse to fixed coords.
     Expected if features-based: predictions stable.

  3. GRAD-CAM on tip keypoint output: where does the model "look"?
     Expected if shortcut: heatmap on background or uniform.
     Expected if features-based: heatmap on actual nozzle tip + spray edges.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "structure"))

from model import AFGNet
from torchvision.transforms.functional import affine

from config import DEVICE, FEAT_H, FEAT_W, IMG_SIZE, OBJ_THRESH


def load_image(path, size=IMG_SIZE):
    """Load and preprocess a single image to [1, 3, H, W]."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size))
    x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return x.unsqueeze(0).to(DEVICE)


def get_best_cell_keypoints(out):
    """Pick the cell with highest centerness*iou and return its keypoints."""
    score = (out["centerness"] * out["iou_quality"])[0, 0]  # [H, W]
    idx = score.argmax()
    i, j = idx // FEAT_W, idx % FEAT_W
    kpts = out["keypoints"][0, :, i, j].detach().cpu().numpy()  # [6]
    return kpts.reshape(3, 2), float(score[i, j].item())


def test_translation_invariance(model, img, shifts=((20, 0), (0, 20), (-20, 0))):
    """Shift image by N pixels, check if keypoint predictions shift accordingly."""
    print("\n=== TEST 1: TRANSLATION INVARIANCE ===")
    model.eval()
    with torch.no_grad():
        out_orig = model(img)
        kpts_orig, _ = get_best_cell_keypoints(out_orig)
        print(
            f"Original tip keypoint:    ({kpts_orig[0, 0]:.4f}, {kpts_orig[0, 1]:.4f})"
        )

        for dx, dy in shifts:
            shifted = affine(img, angle=0, translate=[dx, dy], scale=1.0, shear=[0])
            out = model(shifted)
            kpts, _ = get_best_cell_keypoints(out)
            expected_shift_x = dx / IMG_SIZE
            expected_shift_y = dy / IMG_SIZE
            actual_shift_x = kpts[0, 0] - kpts_orig[0, 0]
            actual_shift_y = kpts[0, 1] - kpts_orig[0, 1]
            print(
                f"  Shift ({dx:+d}px, {dy:+d}px) → tip moved "
                f"({actual_shift_x:+.4f}, {actual_shift_y:+.4f}); "
                f"expected ({expected_shift_x:+.4f}, {expected_shift_y:+.4f})"
            )
            ratio_x = actual_shift_x / (expected_shift_x + 1e-9)
            ratio_y = actual_shift_y / (expected_shift_y + 1e-9)
            print(
                f"    Tracking ratio: x={ratio_x:.2f}, y={ratio_y:.2f}  "
                f"(1.0 = perfect feature tracking, 0.0 = position shortcut)"
            )


def test_background_mask(model, img, kpts_orig):
    """Black out everything outside a tight box around spray, check stability."""
    print("\n=== TEST 2: BACKGROUND MASK ===")
    model.eval()
    H, W = img.shape[-2:]
    cx, cy = int(kpts_orig[0, 0] * W), int(kpts_orig[0, 1] * H)
    pad = 80
    masked = torch.zeros_like(img)
    y1, y2 = max(0, cy - pad), min(H, cy + pad)
    x1, x2 = max(0, cx - pad), min(W, cx + pad)
    masked[:, :, y1:y2, x1:x2] = img[:, :, y1:y2, x1:x2]

    with torch.no_grad():
        out_masked = model(masked)
        kpts_masked, score = get_best_cell_keypoints(out_masked)

    delta = np.linalg.norm(kpts_masked[0] - kpts_orig[0])
    print(f"Tip drift after masking: {delta:.4f} (normalized coords)")
    print(f"  > 0.05 = significant drift = model was using background context")
    print(f"  < 0.01 = stable = model is locating from spray features (good)")


def gradcam_on_tip(model, img, save_path="gradcam_tip.png"):
    """Grad-CAM on tip-x output to see what model attends to."""
    print("\n=== TEST 3: GRAD-CAM ON TIP KEYPOINT ===")
    model.eval()
    features = {}

    def hook(module, inp, out):
        features["fpn"] = out
        out.retain_grad()

    handle = model.fpn.register_forward_hook(hook)

    out = model(img)
    score = (out["centerness"] * out["iou_quality"])[0, 0]
    idx = score.argmax()
    i, j = idx // FEAT_W, idx % FEAT_W
    tip_x = out["keypoints"][0, 0, i, j]
    tip_x.backward()

    fmap = features["fpn"][0]
    grads = features["fpn"].grad[0]
    weights = grads.mean(dim=(1, 2))
    cam = (weights[:, None, None] * fmap).sum(0).clamp(min=0)
    cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()

    handle.remove()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img[0].cpu().permute(1, 2, 0).numpy())
    ax[0].set_title("Input")
    ax[0].axis("off")
    ax[1].imshow(img[0].cpu().permute(1, 2, 0).numpy())
    ax[1].imshow(cam, alpha=0.5, cmap="jet", extent=(0, IMG_SIZE, IMG_SIZE, 0))
    ax[1].set_title("Grad-CAM on tip-x prediction")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved: {save_path}")
    print("  If heatmap on background/uniform → shortcut confirmed")
    print("  If heatmap on nozzle tip → model uses real features")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--img", required=True, help="Path to test spray image")
    args = parser.parse_args()

    model = AFGNet(pretrained=False).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(state if "state_dict" not in state else state["state_dict"])

    img = load_image(args.img)

    with torch.no_grad():
        out = model(img)
    kpts, score = get_best_cell_keypoints(out)
    print(f"Best cell score: {score:.3f}")
    print(f"Predicted keypoints (tip, left, right):\n{kpts}")

    test_translation_invariance(model, img)
    test_background_mask(model, img, kpts)
    gradcam_on_tip(model, img)
