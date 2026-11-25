"""Colab notebook: long LoRA SFT of group A (human-only data) to test the underfitting hypothesis.

Same recipe as the paper (Qwen2.5-1.5B-Instruct, LoRA r16, lr 2e-4, bs 2 x accum 8, max_len 3072, seed 0) but the
cosine schedule spans EPOCHS epochs instead of ~1. Training is cut into one-epoch milestones with --stop_at; after
each milestone the adapter is snapshotted and evaluated (90 prompts + 28 new-category prompts, k=0, connectivity
checker; Gurobi re-check and blind recognition are done locally afterwards) and held-out loss is measured.
Disconnect-safe: all state on Drive, checkpoints every SAVE steps, finished milestones are skipped on Run all.
Code comes from the public repo; data/mix/A.jsonl from MyDrive/brickexp_mix/brickexp_code.zip (built by
tools/build_mix_data.py).
python tools/build_colab_long_notebook.py [--epochs 5] [--limit 0]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--save_steps", type=int, default=40)
ap.add_argument("--limit", type=int, default=0, help="smoke: first N prompts per bucket")
ap.add_argument("--repo", default="https://github.com/lucas-shuo-han/you-get-what-you-verify.git")
ap.add_argument("--out", default=None)
ap.add_argument("--kaggle", action="store_true", help="headless Kaggle kernel: state restored from dataset <user>/brickexp-long-state")
ap.add_argument("--hours", type=float, default=11.3, help="kaggle: stop cleanly before this many hours")
A = ap.parse_args()
if A.out is None:
    A.out = str(ROOT / ("notebooks/kaggle_long_A.ipynb" if A.kaggle else "notebooks/colab_long_A.ipynb"))

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(f"""# Long training of group A ({A.epochs} epochs) — Colab T4
Runtime → T4 GPU → Run all. After a disconnect, just Run all again: finished milestones are skipped and training
resumes from the last checkpoint on Drive (`MyDrive/brickexp_long/`).""")

SETUP = ("""OUTD = "/kaggle/working"; T0 = time.time()
st = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/progress.log", recursive=True)]
if st and not os.path.exists(OUTD + "/progress.log"):   # resume: previous session's state (read-only input)
    shutil.copytree(st[0], OUTD, dirs_exist_ok=True); print("restored state from", st[0])""" if A.kaggle else
"""from google.colab import drive
drive.mount("/content/drive")
OUTD = "/content/drive/MyDrive/brickexp_long"; os.makedirs(OUTD, exist_ok=True); T0 = time.time()""")
code("""import os, sys, glob, json, math, time, zipfile, subprocess, shutil
""" + SETUP + """
def stage(n):
    print(f"=== [{(time.time()-T0)/60:6.1f} min] {n}", flush=True)
    open(f"{OUTD}/progress.log", "a").write(f"{time.ctime()} {n}\\n")
def run_logged(cmd, tag):
    stage(tag)
    p = subprocess.run(f"set -o pipefail; ( {cmd} ) 2>&1 | tee -a {OUTD}/{tag}.log", shell=True, executable="/bin/bash",
                       capture_output=True, text=True)
    print(p.stdout[-3000:])
    if p.returncode != 0:
        open(f"{OUTD}/error.txt", "a").write(f"[{tag}] rc={p.returncode}\\n" + p.stdout[-3000:])
        raise RuntimeError(f"{tag} failed rc={p.returncode}")
    return p
stage("pip")
r = subprocess.run("pip install -q vllm peft datasets networkx bitsandbytes", shell=True, capture_output=True, text=True)
print(r.returncode, r.stderr[-1500:])
subprocess.run("pip uninstall -y -q torchao", shell=True)
print(subprocess.run("pip list 2>/dev/null | grep -iE '^(vllm|torch|transformers|peft) '", shell=True, capture_output=True, text=True).stdout)""")

WK = "/kaggle/temp/RLtheLEGO" if A.kaggle else "/content/RLtheLEGO"
code(f"""WORK = "{WK}"
if not os.path.exists(WORK + "/brickexp"):
    subprocess.run("git clone -q {A.repo} " + WORK, shell=True, check=True)
    subprocess.run("bash tools/fetch_vendor.sh", shell=True, check=True, cwd=WORK)
os.chdir(WORK); sys.path.insert(0, WORK)
ka = glob.glob("/kaggle/input/**/data/mix/A.jsonl", recursive=True)
if ka and not os.path.exists("data/mix/A.jsonl"):
    shutil.copy(ka[0], "data/mix/A.jsonl"); print("A.jsonl from", ka[0])
if not os.path.exists("data/mix/A.jsonl"):
    zs = sorted(glob.glob("/content/drive/MyDrive/**/brickexp_code.zip", recursive=True), key=os.path.getmtime, reverse=True)
    for z in zs:
        with zipfile.ZipFile(z) as f:
            if "data/mix/A.jsonl" in f.namelist():
                f.extract("data/mix/A.jsonl", WORK); print("A.jsonl from", z); break
    assert os.path.exists("data/mix/A.jsonl"), "no brickexp_code.zip with data/mix/A.jsonl found in MyDrive"
nA = sum(1 for _ in open("data/mix/A.jsonl")); print("A rows", nA); assert nA == 4000, nA
assert "--stop_at" in open("brickexp/train_sft.py").read(), "repo too old: train_sft has no --stop_at"
EPOCHS = {A.epochs}
STEP_EPOCH = math.ceil(3789 / 16)          # 3,789 rows survive max_len 3072 (paper run A s0); 1 epoch = 237 steps
TOTAL = STEP_EPOCH * EPOCHS
MILESTONES = [STEP_EPOCH * e for e in range(1, EPOCHS + 1)]
print("total steps", TOTAL, "milestones", MILESTONES)""")

code(f"""def done(p): return os.path.exists(f"{{OUTD}}/{{p}}")
TRAIN = ("export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
         "python -m brickexp.train_sft --model Qwen/Qwen2.5-1.5B-Instruct --data_jsonl data/mix/A.jsonl "
         f"--out {{OUTD}}/sft_A_long --seed 0 --bs 2 --grad_accum 8 --max_len 3072 --rank 16 --lr 2e-4 "
         f"--max_steps {{TOTAL}} --save_steps {A.save_steps} --no_merge")
if not done("probe_ok"):
    pr = run_logged(TRAIN.replace(f"--out {{OUTD}}/sft_A_long", "--out " + WORK + "/probe") + " --probe_steps 3", "probe")
    import re; m = re.search(r"PEAK_MEM_GB ([0-9.]+)", pr.stdout); peak = float(m.group(1)) if m else None
    assert peak is None or peak < 13.5, f"would OOM on T4: {{peak}}"
    open(f"{{OUTD}}/probe_ok", "w").write(str(peak))

def merge(snap, dst):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16, device_map={{"": 0}})
    m = PeftModel.from_pretrained(base, snap).merge_and_unload()
    m.save_pretrained(dst, safe_serialization=True)
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").save_pretrained(dst)
    del m, base; torch.cuda.empty_cache()

EV = "python -m brickexp.eval_2x2 run --backend vllm --max_budget 0 --checker connectivity --limit {A.limit}"
STOPPED = False
for step in MILESTONES:
    tag = f"A_long_e{{step // STEP_EPOCH}}"
    snap = f"{{OUTD}}/snap_{{step}}"
    if not done(f"val_loss_{{step}}.json") and (time.time() - T0) / 3600 + 2.7 > {A.hours}:
        stage(f"TIME GUARD: not starting milestone {{step}}; resume in next session"); STOPPED = True; break
    if not done(f"snap_{{step}}/adapter_model.safetensors"):
        run_logged(TRAIN + f" --stop_at {{step}}", f"train_to_{{step}}")
        ck = f"{{OUTD}}/sft_A_long/ckpt/checkpoint-{{step}}"
        assert os.path.exists(ck), f"missing {{ck}}"
        os.makedirs(snap, exist_ok=True)
        for f in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copy(f"{{ck}}/{{f}}", snap)
        shutil.copy(f"{{OUTD}}/sft_A_long/train_log.json", f"{{snap}}/train_log.json")
    if not done(f"runs/summary_{{tag}}_newcat.json"):
        merged = WORK + "/merged_tmp"; shutil.rmtree(merged, ignore_errors=True); merge(snap, merged)
        for suite, extra, tb in (("90", "", 1.5), ("newcat", " --eval_jsonl data/mix/newcat_test_eval.jsonl", 0.5)):
            if not done(f"runs/summary_{{tag}}_{{suite}}.json"):
                run_logged(EV + f" --out_dir {{OUTD}}/runs --time_budget_h {{tb}} --model {{merged}} --tag {{tag}}_{{suite}}{{extra}}",
                           f"eval_{{tag}}_{{suite}}")
        shutil.rmtree(merged, ignore_errors=True)
    if not done(f"val_loss_{{step}}.json"):
        run_logged(f"python -m brickexp.val_loss --model Qwen/Qwen2.5-1.5B-Instruct --data data/mix/heldout200.jsonl "
                   f"--adapters {{tag}}={{snap}} --max_len 3072 --out {{OUTD}}/val_loss_{{step}}.json", f"valloss_{{step}}")
    s = json.load(open(f"{{OUTD}}/runs/summary_{{tag}}_90.json"))
    stage(f"milestone {{step}} done: " + f"k0 valid {{s['solved_k0']}}/{{s['n']}}")""")

code("""if not STOPPED:
    S = " ".join(sorted(glob.glob(f"{OUTD}/runs/summary_*.json")))
    print(run_logged(f"python -m brickexp.eval_2x2 table {S} --out {OUTD}/TABLE_long.md", "table").stdout)
    stage("ALL DONE")""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
Path(A.out).write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(A.out)
