"""Push the mixed-data kernels (A, B) to Kaggle ONCE each, unattended (no polling). Private, T4, internet on,
checker=connectivity (no secrets). Dataset brickexp-code is uploaded once and shared by both kernels.
Stops on any push error (e.g. quota) and records it in results/DEBUG_LOG.md.
python tools/kaggle_mix_push.py --groups B A
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_push as K  # noqa: E402

ROOT = K.ROOT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--groups", nargs="+", default=["B", "A"])
    ap.add_argument("--timeout_h", type=float, default=11.0)
    a = ap.parse_args()
    user = K.username()
    K.push_dataset(user)
    for g in a.groups:
        slug = f"brickexp-mix-{g.lower()}"
        k = K.STAGE / f"kernel_mix_{g}"; shutil.rmtree(k, ignore_errors=True); k.mkdir(parents=True)
        shutil.copy(ROOT / f"notebooks/kaggle_mix_{g}.ipynb", k / "mix.ipynb")
        (k / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}", "title": slug, "code_file": "mix.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{user}/{K.DS_SLUG}"], "competition_sources": [], "kernel_sources": [],
            "machine_shape": "NvidiaTeslaT4"}, indent=1))
        rc, out = K.sh("kernels", "push", "-p", str(k), "--accelerator", "NvidiaTeslaT4",
                       "-t", str(int(a.timeout_h * 3600)), check=False)
        ok = rc == 0 and "error" not in out.lower()
        print(f"PUSH {slug}: {'ok' if ok else 'FAILED'}", flush=True)
        if not ok:
            with open(ROOT / "results/DEBUG_LOG.md", "a", encoding="utf-8") as f:
                f.write(f"\n- {time.strftime('%Y-%m-%d %H:%M')} kaggle push {slug} FAILED (rc={rc}): {out.strip()[-400:]}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
