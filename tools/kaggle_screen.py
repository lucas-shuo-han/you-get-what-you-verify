"""§4g big-GNN screener on Kaggle GPU (labels are made locally with Gurobi; the license never leaves this machine).

1. zip brickexp/ + vendor brickgpt + data/screen/*.jsonl -> private dataset <user>/brickexp-screen
2. notebook: unzip, run `python -m brickexp.screen.run` with the given args, outputs in /kaggle/working/screen
3. optional wait + fetch to results/screen_kaggle_<tag>/

Usage:
  python tools/kaggle_screen.py --tag smoke --run_args "--seeds 0 --epochs 1 --max_steps 200 --H 192 --L 6"
  python tools/kaggle_screen.py --tag full --skip_dataset --run_args "--seeds 0 1 2 --epochs 15 --H 192 --L 6"
"""
import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_push import sh, username  # noqa: E402  (auth via kaggle.json; never read/printed here)

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "dist" / "kaggle_screen"
DS = "brickexp-screen"
BLOCK = ("gurobi.lic", ".env", "__pycache__", "kaggle.json")


def build_zip():
    d = STAGE / "dataset"; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    with zipfile.ZipFile(d / "screen_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for base in [ROOT / "brickexp", ROOT / "vendor/BrickGPT/src/brickgpt"]:
            for f in base.rglob("*"):
                if f.is_file() and not any(b in str(f) for b in BLOCK):
                    z.write(f, f.relative_to(ROOT))
        for f in sorted((ROOT / "data/screen").glob("*.jsonl")):  # snapshot: complete JSON lines only
            ok = []
            for l in f.read_text().splitlines():
                try:
                    json.loads(l); ok.append(l)
                except ValueError:
                    pass
            z.writestr(str(f.relative_to(ROOT)).replace("\\", "/"), "\n".join(ok) + "\n")
            print(f.name, len(ok), "labels")
        names = z.namelist()
    assert not any(any(b in n for b in BLOCK) for n in names)
    print("zip files:", len(names), (d / "screen_code.zip").stat().st_size // 1024, "KB")
    return d


def push_dataset(user):
    d = build_zip()
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": DS, "id": f"{user}/{DS}", "licenses": [{"name": "other"}]}))
    rc, out = sh("datasets", "status", f"{user}/{DS}", check=False)
    if rc == 0 and "ready" in out.lower():
        sh("datasets", "version", "-p", str(d), "-m", time.strftime("%Y-%m-%d %H:%M"))
    else:
        sh("datasets", "create", "-p", str(d))
    for _ in range(60):
        time.sleep(10)
        _, out = sh("datasets", "status", f"{user}/{DS}", check=False)
        if "ready" in out.lower():
            return
    sys.exit("dataset not ready after 10 min")


def notebook(run_args):
    src = f'''import os, sys, glob, zipfile, subprocess, time, traceback
OUT = "/kaggle/working"
open(f"{{OUT}}/hello.txt", "w").write(time.ctime())
try:
    import shutil
    W = "/kaggle/working/w"
    z = glob.glob("/kaggle/input/**/screen_code.zip", recursive=True)
    if z:
        zipfile.ZipFile(z[0]).extractall(W)
    else:  # Kaggle auto-extracts uploaded zips into a directory (DEBUG_LOG 2026-11-19)
        d = [os.path.dirname(os.path.dirname(os.path.dirname(p)))
             for p in glob.glob("/kaggle/input/**/brickexp/screen/run.py", recursive=True)]
        assert d, "dataset not attached: " + str(glob.glob("/kaggle/input/*/*"))[:500]
        shutil.copytree(d[0], W)
    os.chdir(W)
    # training/eval path is solver-free (needs only torch + numpy); no gurobipy, no license
    assert glob.glob("data/screen/id.jsonl"), "labels missing"
    print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True,
                         capture_output=True, text=True).stdout, flush=True)
    t = time.time()
    p = subprocess.run("set -o pipefail; python -m brickexp.screen.run --data data/screen --out /kaggle/working/screen "
                       + {json.dumps(run_args)} + " 2>&1 | tee /kaggle/working/screen_stdout.log",
                       shell=True, executable="/bin/bash")
    open(f"{{OUT}}/wall_seconds.txt", "w").write(f"{{time.time() - t:.0f}} rc={{p.returncode}}")
    assert p.returncode == 0, f"run failed rc={{p.returncode}}"
except BaseException:
    open(f"{{OUT}}/error.txt", "w").write(traceback.format_exc()); raise
'''
    return {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run_args", required=True)
    ap.add_argument("--skip_dataset", action="store_true")
    ap.add_argument("--timeout_h", type=float, default=3.0)
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    user = username()
    if not a.skip_dataset:
        push_dataset(user)
    slug = f"brickexp-screen-{a.tag}"
    k = STAGE / "kernel"; shutil.rmtree(k, ignore_errors=True); k.mkdir(parents=True)
    json.dump(notebook(a.run_args), open(k / "screen.ipynb", "w"), indent=1)
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "screen.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"{user}/{DS}"], "competition_sources": [], "kernel_sources": [],
        "machine_shape": "NvidiaTeslaT4"}, indent=1))
    sh("kernels", "push", "-p", str(k), "--accelerator", "NvidiaTeslaT4", "-t", str(int(a.timeout_h * 3600)))
    if a.wait:
        t0 = time.time()
        while True:
            time.sleep(60)
            _, out = sh("kernels", "status", f"{user}/{slug}", check=False)
            print(f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
            if any(s in out.lower() for s in ("complete", "error", "cancel")):
                break
        dst = ROOT / "results" / f"screen_kaggle_{a.tag}"; dst.mkdir(parents=True, exist_ok=True)
        sh("kernels", "output", f"{user}/{slug}", "-p", str(dst), "-o", check=False)


if __name__ == "__main__":
    main()
