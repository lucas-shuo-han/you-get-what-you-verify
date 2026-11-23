"""Colab notebook: mixed-data SFT (A = 4000 human, B = human + 3x AI) of Qwen2.5-1.5B-Instruct + eval.

Order (Colab quota may run out): train B -> eval B -> eval base -> train A -> eval A.
Same optimisation steps for A and B: STEPS = ceil(len(B) / (bs*grad_accum)) (= 1 epoch over B, ~1.2 over A).
Disconnect-safe: everything on Drive (OUTD); training checkpoints every SAVE steps and resumes; each stage is skipped
when its output exists. Eval sets: the fixed 90 prompts (k=0 + feedback <= 3) and 28 new-category prompts
(data/mix/newcat_test_eval.jsonl). Checker = connectivity on Colab (as C1); Gurobi re-judge locally.
python tools/build_colab_mix_notebook.py [--limit 0] [--max_steps 0]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--train_args", default="--bs 2 --grad_accum 8 --max_len 3072 --rank 16 --lr 2e-4")
ap.add_argument("--max_steps", type=int, default=0, help="0 = ceil(len(B)/16)")
ap.add_argument("--save_steps", type=int, default=40)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tb90", type=float, default=1.5); ap.add_argument("--tbnew", type=float, default=0.5)
ap.add_argument("--out", default=None)
ap.add_argument("--kaggle", action="store_true", help="unattended Kaggle kernel for ONE group (--group)")
ap.add_argument("--group", default="B", choices=["A", "B", "C", "FA", "FB", "FC"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mult", type=int, default=1, help="step multiplier (training-length check)")
ap.add_argument("--max_budget", type=int, default=3, help="0 = k=0 only")
ap.add_argument("--with_base", action="store_true", help="kaggle: also eval the untrained base at the end")
A = ap.parse_args()
if A.out is None:
    A.out = str(ROOT / ("notebooks/kaggle_mix_%s.ipynb" % A.group if A.kaggle else "notebooks/colab_mix_sft.ipynb"))
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Mixed-data SFT (A: 4000 human, B: + 3x AI) + eval — Colab free T4
Runtime → GPU (T4). Put `brickexp_code.zip` in `MyDrive/brickexp_mix/` (or upload to /content). Run all; after a
disconnect just Run all again — finished stages are skipped and training resumes from its last checkpoint.
Order: train B → eval B → eval base → train A → eval A. Checker = connectivity (as C1); Gurobi re-judge locally.""")

SETUP = ('OUTD = "/kaggle/working"' if A.kaggle else
         'from google.colab import drive\ndrive.mount("/content/drive")\nOUTD = "/content/drive/MyDrive/brickexp_mix"')
code("""import os, sys, glob, json, math, time, zipfile, subprocess
""" + SETUP + """
os.makedirs(OUTD, exist_ok=True); T0 = time.time()
def stage(n):
    print(f"=== [{(time.time()-T0)/60:6.1f} min] {n}", flush=True)
    open(f"{OUTD}/stage_{time.strftime('%m%d_%H%M%S')}_{n.replace(' ', '_')}.txt", "w").write(time.ctime())
def run_logged(cmd, tag):
    stage(tag)
    p = subprocess.run(f"set -o pipefail; ( {cmd} ) 2>&1 | tee -a {OUTD}/{tag}.log", shell=True, executable="/bin/bash",
                       capture_output=True, text=True)
    print(p.stdout[-4000:])
    if p.returncode != 0:
        open(f"{OUTD}/error.txt", "a").write(f"[{tag}] rc={p.returncode}\\n" + p.stdout[-3000:])
        raise RuntimeError(f"{tag} failed rc={p.returncode}")
    return p
stage("pip")
r = subprocess.run("pip install -q vllm peft gurobipy datasets networkx bitsandbytes", shell=True, capture_output=True, text=True)
print(r.returncode, r.stderr[-1500:])
subprocess.run("pip uninstall -y -q torchao", shell=True)
print(subprocess.run("pip list 2>/dev/null | grep -iE '^(vllm|torch|transformers|peft) '", shell=True, capture_output=True, text=True).stdout)""")

WORKDIR = '"/kaggle/temp/RLtheLEGO"' if A.kaggle else '"/content/RLtheLEGO"'
ZIPS = ('glob.glob("/kaggle/input/**/brickexp_code.zip", recursive=True)' if A.kaggle else
        'glob.glob(OUTD + "/brickexp_code.zip") + glob.glob("/content/**/brickexp_code.zip", recursive=True)')
code("""WORK = """ + WORKDIR + """  # code on local disk, outputs in OUTD
zips = """ + ZIPS + """
if not os.path.exists(WORK + "/brickexp"):
    dirs = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/brickexp/__init__.py", recursive=True)]
    if zips:
        zipfile.ZipFile(zips[0]).extractall(WORK)
    elif dirs:
        import shutil; os.makedirs(os.path.dirname(WORK), exist_ok=True); shutil.copytree(dirs[0], WORK, dirs_exist_ok=True)
    else:
        raise RuntimeError("brickexp code not found (zip or extracted dataset)")
os.chdir(WORK); sys.path.insert(0, WORK)
man = json.load(open("data/mix/manifest.json")); print(man)
nB = sum(1 for _ in open("data/mix/B.jsonl")); nA = sum(1 for _ in open("data/mix/A.jsonl"))
STEPS = (""" + (str(A.max_steps) if A.max_steps else "math.ceil(nB / 16)") + """) * """ + str(A.mult) + """
print("A", nA, "B", nB, "optimizer steps per group:", STEPS)""")

EV = (f"python -m brickexp.eval_2x2 run --backend vllm --max_budget {A.max_budget} --checker connectivity --limit {A.limit} "
      f"--out_dir {{OUTD}}/runs")
code("""def done(p): return os.path.exists(f"{OUTD}/{p}")
def train(g):
    if done(f"sft_{g}/train_log.json") and (done(f"sft_{g}/merged/config.json") or done(f"runs/summary_{g}_newcat.json")):
        print(f"train {g}: done, skipping"); return
    cmd = (f"export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
           f"python -m brickexp.train_sft --model """ + A.model + """ --data_jsonl data/mix/{g.split('_')[0]}.jsonl --out {OUTD}/sft_{g} --seed """ + str(A.seed) + """ "
           f\"""" + A.train_args + f""" --max_steps {{STEPS}} --save_steps """ + str(A.save_steps) + """")
    if not done(f"sft_{g}/probe_ok"):
        pr = run_logged(cmd + " --probe_steps 3 --no_merge", f"train_{g}_probe")
        import re; m = re.search(r"PEAK_MEM_GB ([0-9.]+)", pr.stdout); peak = float(m.group(1)) if m else None
        assert peak is None or peak < 13.5, f"would OOM on T4: {peak}"
        os.makedirs(f"{OUTD}/sft_{g}", exist_ok=True); open(f"{OUTD}/sft_{g}/probe_ok", "w").write(str(peak))
    run_logged(cmd, f"train_{g}")
    L = json.load(open(f"{OUTD}/sft_{g}/train_log.json"))
    print("loss:", [(h.get("step"), round(h["loss"], 4)) for h in L["log_history"] if "loss" in h][-10:])
def evaluate(tag, model):
    for suite, extra, tb in (("90", "", """ + str(A.tb90) + """), ("newcat", " --eval_jsonl data/mix/newcat_test_eval.jsonl", """ + str(A.tbnew) + """)):
        if done(f"runs/summary_{tag}_{suite}.json"):
            print(f"eval {tag} {suite}: done, skipping"); continue
        run_logged(f\"""" + EV + """ --time_budget_h {tb} --model {model} --tag {tag}_{suite}{extra}", f"eval_{tag}_{suite}")""")

GROUPS = [f"{A.group}_s{A.seed}_x{A.mult}"] if A.kaggle else ["B", "base", "A"]
if A.kaggle and A.with_base:
    GROUPS.append("base")
for g in GROUPS:
    if g == "base":
        code(f"""try:
    evaluate("base", "{A.model}")
except Exception as e:
    print("base eval failed:", e)  # never lose the group results because of the reference row""")
    else:
        code(f"""train("{g}"); evaluate("{g}", f"{{OUTD}}/sft_{g}/merged")
if done("runs/summary_{g}_newcat.json"): subprocess.run(f"rm -rf {{OUTD}}/sft_{g}/merged {{OUTD}}/sft_{g}/ckpt", shell=True)""")
code("""S = " ".join(sorted(p.replace(chr(92), "/") for p in glob.glob(f"{OUTD}/runs/summary_*.json")))
print(run_logged(f"python -m brickexp.eval_2x2 table {S} --out {OUTD}/TABLE_mix.md", "table").stdout)
stage("done")""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
Path(A.out).write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(A.out)
