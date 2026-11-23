"""Generates notebooks/kaggle_distill_sft.ipynb: LoRA SFT of Qwen2.5-1.5B-Instruct on distilled DeepSeek data, then
the 2x2 eval ({base, sft} x {k=0, feedback<=3}) on the fixed 90 prompts. Thin shell: logic lives in brickexp/.

Kaggle inputs (private datasets): dist/brickexp_code.zip (python tools/make_kaggle_bundle.py; it ships
data/distill/sft_distill.jsonl made by tools/distill_to_sft.py). Accelerator: GPU T4 (fp16). Internet on.
Optional Secrets GRB_WLSACCESSID / GRB_WLSSECRET / GRB_LICENSEID for the Gurobi stability check.
Also runs on Colab (set OUTD; secrets cell then falls back to connectivity unless GRB_LICENSE_FILE is set).

python tools/build_distill_notebook.py                       # full run
python tools/build_distill_notebook.py --limit 2 --max_steps 5 --out notebooks/kaggle_distill_smoke.ipynb
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--data", default="data/distill/sft_distill.jsonl")
ap.add_argument("--train_args", default="--epochs 2 --bs 2 --grad_accum 8 --max_len 3072 --rank 16 --lr 2e-4")
ap.add_argument("--max_steps", type=int, default=-1)
ap.add_argument("--checker", default="gurobi", choices=["gurobi", "auto", "connectivity"])
ap.add_argument("--max_budget", type=int, default=3)
ap.add_argument("--limit", type=int, default=0, help="prompts per bucket (0 = all 30)")
ap.add_argument("--skip_base", action="store_true", help="skip the untrained-base row")
ap.add_argument("--time_budget_h", type=float, default=3.5, help="per eval run")
ap.add_argument("--out", default=None)
A = ap.parse_args()

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md(f"""# Distill SFT: {A.model} on DeepSeek-distilled, verify-filtered bricks + 2x2 eval
Accelerator **GPU T4** (fp16), Internet on. Attach the private dataset holding `brickexp_code.zip`.
Outputs in /kaggle/working: `sft/adapter`, `sft/train_log.json` (loss), `runs/eval_*.jsonl`, `runs/summary_*.json`, `TABLE_2x2.md`.""")

code("""import os, sys, glob, json, time, zipfile, subprocess, traceback
OUTD = "/kaggle/working" if os.path.exists("/kaggle") else "/content/work"
os.makedirs(OUTD, exist_ok=True); T0 = time.time()
def stage(n):
    print(f"=== [{(time.time()-T0)/60:6.1f} min] {n}", flush=True)
    open(f"{OUTD}/stage_{int(time.time()-T0):06d}_{n.replace(' ', '_')}.txt", "w").write(time.ctime())
def run_logged(cmd, tag):
    stage(tag)
    p = subprocess.run(f"set -o pipefail; ( {cmd} ) 2>&1 | tee {OUTD}/{tag}.log", shell=True, executable="/bin/bash",
                       capture_output=True, text=True)
    print(p.stdout[-4000:])
    if p.returncode != 0:
        open(f"{OUTD}/error.txt", "a").write(f"[{tag}] rc={p.returncode}\\n" + p.stdout[-3000:])
        raise RuntimeError(f"{tag} failed rc={p.returncode}")
    return p
stage("pip")
r = subprocess.run("pip install -q vllm peft gurobipy datasets networkx bitsandbytes", shell=True, capture_output=True, text=True)
print(r.returncode, r.stderr[-1500:])
subprocess.run("pip uninstall -y -q torchao", shell=True)  # old torchao on Kaggle breaks recent peft import
print(subprocess.run("pip list 2>/dev/null | grep -iE '^(vllm|torch|transformers|peft) '", shell=True, capture_output=True, text=True).stdout)""")

code("""WORK = OUTD + "/RLtheLEGO"
zips = glob.glob("/kaggle/input/**/brickexp_code.zip", recursive=True) + glob.glob("/content/**/brickexp_code.zip", recursive=True)
if not os.path.exists(WORK + "/brickexp"):
    assert zips, "attach the dataset containing brickexp_code.zip"
    zipfile.ZipFile(zips[0]).extractall(WORK)
os.chdir(WORK); sys.path.insert(0, WORK)
print(sorted(os.listdir(WORK)))""")

code(f"""# Gurobi WLS license from Kaggle Secrets (never printed); written to /tmp, not the saved output dir
try:
    from kaggle_secrets import UserSecretsClient
    sc = UserSecretsClient()
    vals = {{k: sc.get_secret(k) for k in ("GRB_WLSACCESSID", "GRB_WLSSECRET", "GRB_LICENSEID")}}
    open("/tmp/gurobi.lic", "w").write("".join(f"{{k[4:]}}={{v}}\\n" for k, v in vals.items())); del vals
    os.environ["GRB_LICENSE_FILE"] = "/tmp/gurobi.lic"
except Exception as e:
    print("no Gurobi secrets:", type(e).__name__)
from brickexp import checker
OK = checker.gurobi_available(); print("GUROBI SELF-CHECK:", "OK" if OK else "FAILED")
if "{A.checker}" == "gurobi" and not OK:
    raise checker.GurobiUnavailable("checker=gurobi but no working license; rebuild with --checker auto to fall back")
n = sum(1 for _ in open("{A.data}")); print("SFT examples:", n); assert n > 0""")

code(f"""# ---- LoRA SFT, fp16 on one T4 (subprocess so the GPU is free for vLLM afterwards) ----
TRAIN = (f"export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
         f"python -m brickexp.train_sft --model {A.model} --data_jsonl {A.data} --out {{OUTD}}/sft "
         f"{A.train_args} --max_steps {A.max_steps}")
pr = run_logged(TRAIN + " --probe_steps 5 --no_merge", "train_probe")   # memory probe before the full run
import re; m = re.search(r"PEAK_MEM_GB ([0-9.]+)", pr.stdout); PEAK = float(m.group(1)) if m else None
print("probe peak GB", PEAK); assert PEAK is not None and PEAK < 13.5, "would OOM on T4 (15 GB)"
run_logged(TRAIN, "train_full")
subprocess.run(f"rm -rf {{OUTD}}/sft/ckpt", shell=True)
L = json.load(open(f"{{OUTD}}/sft/train_log.json"))
print("loss:", [(h.get("step"), round(h["loss"], 4)) for h in L["log_history"] if "loss" in h])""")

EV = (f"python -m brickexp.eval_2x2 run --backend vllm --max_budget {A.max_budget} --checker {A.checker} "
      f"--limit {A.limit} --time_budget_h {A.time_budget_h} --out_dir {{OUTD}}/runs")
runs = [] if A.skip_base else [("base", A.model)]
runs.append(("sft", "{OUTD}/sft/merged"))
for tag, model in runs:
    code(f"""run_logged(f"{EV} --model {model} --tag {tag}", "eval_{tag}")""")

code(f"""S = " ".join(f"{{OUTD}}/runs/summary_{{t}}.json" for t in {[t for t, _ in runs]!r})
print(run_logged(f"python -m brickexp.eval_2x2 table {{S}} --out {{OUTD}}/TABLE_2x2.md", "table").stdout)
subprocess.run(f"rm -rf {{OUTD}}/sft/merged {{OUTD}}/RLtheLEGO", shell=True)  # keep adapter + logs only
stage("done")""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
p = Path(A.out) if A.out else Path(__file__).resolve().parent.parent / "notebooks" / "kaggle_distill_sft.ipynb"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(p)
