"""CS182 paper top-up kernel (one Kaggle T4 run, unattended):
  1. held-out validation loss (data/mix/heldout200.jsonl, 200 train-split rows unused by any group / eval) for the
     downloaded adapters A_s0, B_s0, B_s1, C_s0 (+ base row)
  2. train A seed 1 (same config as A_s0: data/mix/A.jsonl, 304 steps) -> eval the 90 prompts k=0 (+ new categories)
  3. validation loss of the new A_s1 adapter
Adapters are uploaded once as private dataset <user>/brickexp-adapters.
python tools/kaggle_paper_push.py [--build_only]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_push as K  # noqa: E402

ROOT = K.ROOT
ADAPTERS = {"A_s0": "results/mix_A/sft_A/adapter", "B_s0": "results/mix_B/sft_B/adapter",
            "B_s1": "results/final_small/b1x1/sft_B_s1_x1/adapter", "C_s0": "results/final_small/c0x1/sft_C_s0_x1/adapter"}
DS_AD = "brickexp-adapters"
NB = ROOT / "notebooks/final/paper_A_s1_valloss.ipynb"
VL = '''ADAPT_ROOT = "/kaggle/input"
def find_adapters(names):
    out = {}
    for n in names:
        c = [os.path.dirname(p).replace(chr(92), "/") for p in glob.glob(f"{ADAPT_ROOT}/**/{n}/adapter_config.json", recursive=True)]
        assert c, f"adapter {n} not attached"
        out[n] = c[0]
    return out
def val_loss(names_dirs, out_name):
    if done(out_name): print(out_name, "done, skipping"); return
    spec = " ".join(f"{k}={v}" for k, v in names_dirs.items())
    run_logged(f"python -m brickexp.val_loss --model Qwen/Qwen2.5-1.5B-Instruct --data data/mix/heldout200.jsonl "
               f"--adapters {spec} --out {OUTD}/{out_name}", out_name.replace(".json", ""))
'''


def build():
    subprocess.run([sys.executable, str(ROOT / "tools/build_colab_mix_notebook.py"), "--kaggle", "--group", "A", "--seed", "1",
                    "--mult", "1", "--max_budget", "0", "--max_steps", "304", "--tb90", "2", "--tbnew", "1", "--out", str(NB)],
                   check=True)
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    i = next(k for k, c in enumerate(cells) if 'train("A_s1_x1")' in "".join(c["source"]))
    mk = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    cells.insert(i, mk(VL + 'try:\n    val_loss(find_adapters(%r), "val_loss_4adapters.json")\n'
                       'except Exception as e:\n    print("val loss (4 adapters) failed:", e)  # never block A_s1\n'
                       % list(ADAPTERS)))
    cells.insert(i + 2, mk('try:\n    val_loss({"A_s1": f"{OUTD}/sft_A_s1_x1/adapter"}, "val_loss_A_s1.json")\n'
                           'except Exception as e:\n    print("val loss A_s1 failed:", e)\n'))
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(NB)


def push_adapters(user):
    d = K.STAGE / "adapters_ds"; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    for n, p in ADAPTERS.items():
        shutil.copytree(ROOT / p, d / n)
    (d / "dataset-metadata.json").write_text(json.dumps({"title": DS_AD, "id": f"{user}/{DS_AD}", "licenses": [{"name": "other"}]}))
    rc, out = K.sh("datasets", "status", f"{user}/{DS_AD}", check=False)
    if rc == 0 and "ready" in out.lower():
        K.sh("datasets", "version", "-p", str(d), "-m", "adapters", "--dir-mode", "zip")
    else:
        K.sh("datasets", "create", "-p", str(d), "--dir-mode", "zip")
    import time
    for _ in range(90):
        time.sleep(10)
        if "ready" in K.sh("datasets", "status", f"{user}/{DS_AD}", check=False)[1].lower():
            return
    sys.exit("adapters dataset not ready")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build_only", action="store_true"); a = ap.parse_args()
    build()
    if a.build_only:
        return
    user = K.username()
    K.push_dataset(user)
    push_adapters(user)
    slug = "brickexp-paper-as1"
    k = K.STAGE / "kernel_paper"; shutil.rmtree(k, ignore_errors=True); k.mkdir(parents=True)
    shutil.copy(NB, k / "run.ipynb")
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "run.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/{K.DS_SLUG}", f"{user}/{DS_AD}"], "competition_sources": [], "kernel_sources": [],
        "machine_shape": "NvidiaTeslaT4"}, indent=1))
    rc, out = K.sh("kernels", "push", "-p", str(k), "--accelerator", "NvidiaTeslaT4", "-t", str(11 * 3600), check=False)
    print("PUSH", slug, "ok" if rc == 0 and "error" not in out.lower() else "FAILED: " + out.strip()[-300:])


if __name__ == "__main__":
    main()
