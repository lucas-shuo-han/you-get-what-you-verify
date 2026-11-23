"""Zip brickexp/ + vendor BrickGPT package (+ fixed eval splits) for upload as a private Kaggle Dataset.
Never includes licenses/secrets. Usage: python tools/make_kaggle_bundle.py -> dist/brickexp_code.zip"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "brickexp_code.zip"
INCLUDE = [ROOT / "brickexp", ROOT / "tests", ROOT / "vendor/BrickGPT/src/brickgpt", ROOT / "data/splits", ROOT / "data/tiny", ROOT / "data/distill", ROOT / "data/mix"]  # data/distill: SFT jsonl from tools/distill_to_sft.py
BLOCK = ("gurobi.lic", ".env", "__pycache__")

OUT.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for base in [b for b in INCLUDE if b.exists()]:
        for f in base.rglob("*"):
            if f.is_file() and not any(b in str(f) for b in BLOCK):
                z.write(f, f.relative_to(ROOT))
    z.write(ROOT / "pytest.ini", "pytest.ini")
print(OUT, OUT.stat().st_size // 1024, "KB")
