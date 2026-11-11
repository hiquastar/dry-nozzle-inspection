"""
gate_forward_snapshot.py - bit-exact gate for code changes that must not alter
the final model.  Run BEFORE a change with --save, and AFTER it with --check:
the forward outputs of the final checkpoint on fixed validation frames must be
identical (max |diff| == 0) when the new switch is off.

  python evaluation/gate_forward_snapshot.py --save  results/gate_snapshot.pt
  python evaluation/gate_forward_snapshot.py --check results/gate_snapshot.pt

Env must match the final model: AFGNET_INSTANCE_NORM=0 AFGNET_TIP_RELATIVE_EDGES=1
AFGNET_MIST_HEAD=1 AFGNET_LEARNED_HEADS=1
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "structure"))
from config import DEVICE, VAL_JSON  # noqa: E402
from dataset import AFGNetDataset, afgnet_collate  # noqa: E402
from model import AFGNet  # noqa: E402

CKPT = HERE.parent / "checkpoint" / "abl_heads_s42" / "afgnet_v10.pth"


def outputs():
    torch.manual_seed(0)
    ds = AFGNetDataset(VAL_JSON, augment=False)
    imgs, targets = afgnet_collate([ds[i] for i in (0, 7, 15, 30)])
    m = AFGNet(pretrained=False).to(DEVICE)
    sd = torch.load(CKPT, map_location=DEVICE)
    m.load_state_dict(sd["model"] if "model" in sd else sd, strict=True)
    m.eval()
    inst = {k: targets[k].to(DEVICE) for k in ("inst_tip", "inst_axis", "inst_valid")}
    with torch.no_grad():
        out = m(imgs.to(DEVICE), inst=inst)
    return {k: v.detach().float().cpu() for k, v in out.items() if torch.is_tensor(v)}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save"); g.add_argument("--check")
    a = ap.parse_args()
    out = outputs()
    if a.save:
        torch.save(out, a.save); print("saved", a.save, sorted(out))
        return
    ref = torch.load(a.check)
    worst = 0.0
    for k in sorted(set(ref) | set(out)):
        if k not in ref or k not in out:
            print(f"  {k:14s} present only in {'reference' if k in ref else 'new'}")
            continue
        d = (ref[k] - out[k]).abs().max().item(); worst = max(worst, d)
        print(f"  {k:14s} max|diff| = {d:.3e}")
    print("GATE", "PASS (bit-exact)" if worst == 0.0 else f"FAIL (max diff {worst:.3e})")


if __name__ == "__main__":
    main()
