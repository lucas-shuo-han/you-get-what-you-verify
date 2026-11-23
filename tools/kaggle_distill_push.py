"""Push the distill-SFT 2x2 notebook to Kaggle ONCE (private, T4, internet on), then optionally wait + fetch.
Reuses tools/kaggle_push.py auth/dataset helpers (credentials never read or printed here).
python tools/kaggle_distill_push.py --limit 10 [--wait]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_push as K  # noqa: E402

ROOT = K.ROOT
SLUG = "brickexp-distill-sft"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10); ap.add_argument("--time_budget_h", type=float, default=1.5)
    ap.add_argument("--timeout_h", type=float, default=5.0); ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    user = K.username()
    K.push_dataset(user)
    k = K.STAGE / "kernel_distill"; shutil.rmtree(k, ignore_errors=True); k.mkdir(parents=True)
    nb = k / "distill_sft.ipynb"
    subprocess.run([sys.executable, str(ROOT / "tools/build_distill_notebook.py"), "--limit", str(a.limit),
                    "--time_budget_h", str(a.time_budget_h), "--out", str(nb)], check=True)
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{SLUG}", "title": SLUG, "code_file": nb.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{user}/{K.DS_SLUG}"],
        "competition_sources": [], "kernel_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=1))
    K.sh("kernels", "push", "-p", str(k), "--accelerator", "NvidiaTeslaT4", "-t", str(int(a.timeout_h * 3600)))
    if a.wait:
        t0 = time.time()
        while True:
            time.sleep(120)
            _, out = K.sh("kernels", "status", f"{user}/{SLUG}", check=False)
            print(f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
            if any(s in out.lower() for s in ("complete", "error", "cancel")):
                break
        dst = ROOT / "results/distill_sft"; dst.mkdir(parents=True, exist_ok=True)
        K.sh("kernels", "output", f"{user}/{SLUG}", "-p", str(dst), "-o", check=False)


if __name__ == "__main__":
    main()
