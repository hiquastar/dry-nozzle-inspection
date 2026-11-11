"""
zip_pi5.py - pack deployment/pi5_final into deployment/pi5_final.zip for the
Raspberry Pi 5 (USB stick). Leaves out caches, virtual environments and old
results; the scripts keep the Unix execute bit, so run.sh works after unzip.

    python deploy/zip_pi5.py            (from train/model_2)
"""

import zipfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[3] / "deployment" / "pi5_final"
SKIP = {"__pycache__", ".venv", "results"}
EXEC = {"run.sh", "benchmark.py"}


def main():
    out = PKG.with_suffix(".zip")
    files = sorted(p for p in PKG.rglob("*") if p.is_file() and not SKIP & set(p.relative_to(PKG).parts))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            info = zipfile.ZipInfo.from_file(p, arcname=str(Path(PKG.name) / p.relative_to(PKG)).replace("\\", "/"))
            info.external_attr = ((0o100755 if p.name in EXEC else 0o100644) & 0xFFFF) << 16
            info.create_system = 3  # Unix, so unzip on the Pi applies the permissions
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    for p in files:
        print(f"  {p.relative_to(PKG)}  {p.stat().st_size / 1e6:.2f} MB")
    print(f"wrote {out}  {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
