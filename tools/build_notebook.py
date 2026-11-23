"""Generates notebooks/kaggle_c0_budget_curve.ipynb (thin shell; all logic lives in brickexp/)."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--per_bucket", type=int, default=5)
ap.add_argument("--max_budget", type=int, default=10)
ap.add_argument("--checker", default="connectivity")
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--out", default=None)
ap.add_argument("--tag", default="c0")
ap.add_argument("--eval_set", type=int, default=10, help="which shipped split: test_eval_{10|100}_s0")
ap.add_argument("--time_budget_h", type=float, default=8.0)
ap.add_argument("--max_tokens", type=int, default=3072)
ap.add_argument("--allow_hf", action="store_true")
ap.add_argument("--vllm_spec", default="vllm", help='e.g. "vllm==0.17.1"; "none" = skip vLLM install')
ap.add_argument("--enforce_eager", action="store_true")
ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
ap.add_argument("--hf_batch", type=int, default=32)
ap.add_argument("--hf_adapter_base", default="", help="load this ungated base, then the --model LoRA adapter on top")
ap.add_argument("--fallback_models", nargs="*", default=["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-1.7B"])
ap.add_argument("--merged_glob", default="", help="eval an already-merged model found under /kaggle/input (glob for its dir)")
ap.add_argument("--train_args", default="", help="if set: run brickexp.train_sft with these args first, eval merged model")
A = ap.parse_args()

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# C0 budget curve (untrained Qwen) — Kaggle shell
Setup: **Accelerator = GPU T4 x2**, Internet on.
1. Upload `dist/brickexp_code.zip` (from `python tools/make_kaggle_bundle.py`) as a **private Kaggle Dataset** and attach it, *or* set `GIT_URL` below.
2. Optional (mechanics checker): add Kaggle **Secrets** `GRB_WLSACCESSID`, `GRB_WLSSECRET`, `GRB_LICENSEID`. Without them the checker falls back to connectivity.
The license file is written to `/tmp` (not `/kaggle/working`, which is saved as output) and is never printed.""")

code("""# ---- observability: output dir check, heartbeat files, tracebacks to file ----
import os, sys, time, traceback, datetime
OUTD = "/kaggle/working"
os.makedirs(OUTD, exist_ok=True)
open(f"{OUTD}/hello.txt", "w").write(f"hello {datetime.datetime.utcnow().isoformat()} cwd={os.getcwd()}\\n")
_RUNLOG = open(f"{OUTD}/run.log", "a", buffering=1)
class _Tee:
    def __init__(self, s): self.s = s
    def write(self, x): self.s.write(x); _RUNLOG.write(x)
    def flush(self): self.s.flush(); _RUNLOG.flush()
    def __getattr__(self, k): return getattr(self.s, k)
sys.stdout, sys.stderr = _Tee(sys.stdout), _Tee(sys.stderr)
def _on_error(shell, etype, evalue, tb, tb_offset=None):
    msg = "".join(traceback.format_exception(etype, evalue, tb))
    open(f"{OUTD}/error.txt", "a").write(msg)
    _RUNLOG.write(msg); _RUNLOG.flush()
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
    os._exit(1)  # stop the run: later cells must not run on broken state
get_ipython().set_custom_exc((BaseException,), _on_error)
print("observability on; hello.txt written")""")

code(f"""import time, subprocess; T0 = time.time()
def stage(name):
    print(f"=== [{{(time.time()-T0)/60:6.1f}} min] {{name}}", flush=True)
    open(f"/kaggle/working/stage_{{int(time.time()-T0):06d}}_{{name.replace(' ', '_')}}.txt", "w").write(time.ctime())
stage("pip install")
r = subprocess.run("pip install -q " + ("" if "{A.vllm_spec}" == "none" else "{A.vllm_spec} nvidia-cutlass-dsl ") + "gurobipy datasets networkx peft", shell=True, capture_output=True, text=True)
# Kaggle ships torchao 0.10; recent peft refuses to import with it (optional dep) -> remove it
subprocess.run("pip uninstall -y -q torchao", shell=True)
print(r.returncode, r.stderr[-1500:])
r = subprocess.run("pip list 2>/dev/null | grep -iE '^(vllm|torch|transformers) '", shell=True, capture_output=True, text=True); print(r.stdout)
stage("pip done")
# flashinfer JIT links with -lcuda; Kaggle images only ship libcuda.so.1 -> add the dev symlink
r = subprocess.run("for d in /usr/lib/x86_64-linux-gnu /usr/local/nvidia/lib64 /usr/lib64; do "
                   "  if [ -e $d/libcuda.so.1 ]; then ln -sf $d/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so; "
                   "  ln -sf $d/libcuda.so.1 /usr/local/cuda/lib64/stubs/libcuda.so 2>/dev/null; echo linked $d; break; fi; done; "
                   "ls -l /usr/local/cuda/lib64/libcuda.so", shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr[-300:])
import os
os.environ["LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:" + os.environ.get("LIBRARY_PATH", "")""")

code("""import os, sys, glob, zipfile, subprocess
GIT_URL = ""  # optional: a git repo containing brickexp/ and vendor/BrickGPT
WORK = "/kaggle/working/RLtheLEGO"
zips = glob.glob("/kaggle/input/**/brickexp_code.zip", recursive=True)
dirs = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/brickexp/__init__.py", recursive=True)]
if not os.path.exists(WORK + "/brickexp"):
    if zips:
        zipfile.ZipFile(zips[0]).extractall(WORK)
    elif dirs:
        subprocess.run(["cp", "-r", dirs[0], WORK], check=True)
    elif GIT_URL:
        subprocess.run(["git", "clone", "--recursive", GIT_URL, WORK], check=True)
    else:
        raise RuntimeError("Attach the brickexp_code dataset or set GIT_URL")
os.chdir(WORK); sys.path.insert(0, WORK)
print(sorted(os.listdir(WORK)))""")

code("""# Gurobi WLS license from Kaggle Secrets (values never printed)
try:
    from kaggle_secrets import UserSecretsClient
    sc = UserSecretsClient()
    vals = {k: sc.get_secret(k) for k in ("GRB_WLSACCESSID", "GRB_WLSSECRET", "GRB_LICENSEID")}
    with open("/tmp/gurobi.lic", "w") as f:
        f.write("".join(f"{k[4:]}={v}\\n" for k, v in vals.items()))
    os.environ["GRB_LICENSE_FILE"] = "/tmp/gurobi.lic"
    del vals
    print("Gurobi secrets loaded")
except Exception as e:
    print("No Gurobi secrets -> connectivity fallback:", type(e).__name__)
from brickexp import checker
# self-check: empty-ish model, solve once (prints only success/failure, never secret values)
ok = checker.gurobi_available()
print("GUROBI SELF-CHECK:", "OK" if ok else "FAILED")""")
code(f"""if "{A.checker}" == "gurobi" and not ok:
    raise checker.GurobiUnavailable("CHECKER=gurobi but Gurobi self-check failed (secrets not attached to this kernel version?)")""")

code("""!python -m pytest -q tests 2>&1 | tail -3""")

if A.train_args:
    code(f"""# ---- LoRA SFT (subprocess, frees GPU before vLLM) ----
import subprocess, time
t = time.time()
# vLLM's pinned transformers predates qwen3_5 -> train with a newer transformers in an isolated dir
pr = subprocess.run("pip install -q --target /kaggle/tmp/newtf --no-deps 'transformers>=5.2' peft accelerate 'huggingface_hub>=1.0' tokenizers safetensors bitsandbytes", shell=True, capture_output=True, text=True)
print("newtf install rc", pr.returncode, pr.stderr[-500:])
import os
env_pp = "/kaggle/tmp/newtf:" + os.environ.get("PYTHONPATH", "") + ":" + os.getcwd()
TRAIN_CMD = f"export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; PYTHONPATH={{env_pp}} python -c 'import transformers;print(transformers.__version__)' && PYTHONPATH={{env_pp}} python -m brickexp.train_sft --model {A.model} --out /kaggle/working/{A.tag} {A.train_args}"
# 20-step memory probe first: abort before burning a full run if peak >= 12 GB
def run_logged(cmd, tag):
    # run with output teed to /kaggle/working/<tag>.log; on failure write error.txt and raise
    stage(tag)
    p = subprocess.run(f"set -o pipefail; ( {{cmd}} ) 2>&1 | tee /kaggle/working/{{tag}}.log", shell=True,
                       executable="/bin/bash", capture_output=True, text=True)
    print(p.stdout[-4000:])
    if p.returncode != 0:
        open("/kaggle/working/error.txt", "a").write(f"[{{tag}}] rc={{p.returncode}}\\n" + p.stdout[-3000:])
        raise RuntimeError(f"{{tag}} failed rc={{p.returncode}}")
    return p
pr = run_logged(TRAIN_CMD + " --probe_steps 20", "train_probe")
import re as _re
_m = _re.search(r"PEAK_MEM_GB ([0-9.]+)", pr.stdout)
PEAK = float(_m.group(1)) if _m else None
print("PROBE peak GB:", PEAK)
if PEAK is None or PEAK >= 12:
    open("/kaggle/working/error.txt", "a").write(f"memory probe: peak={{PEAK}}\\n")
    raise RuntimeError(f"memory probe failed: peak={{PEAK}}")
r = run_logged(TRAIN_CMD, "train_full")
print("train exit", r.returncode, f"{{(time.time()-t)/60:.1f}} min")
assert r.returncode == 0
!rm -rf /kaggle/working/{A.tag}/ckpt
!cat /kaggle/working/{A.tag}/train_log.json | head -c 1500""")
if A.train_args or A.merged_glob:
    # merged text-only Qwen3_5ForCausalLM + transformers-5 tokenizer: vLLM 0.17.1 can load neither -> rebase onto
    # the base repo layout (DEBUG_LOG 2026-11-18, C1 v11 eval failure)
    SRC = f'os.path.dirname(sorted(glob.glob("{A.merged_glob}", recursive=True))[0])' if A.merged_glob else f'"/kaggle/working/{A.tag}/merged"'
    code(f"""import glob, subprocess, json
stage("rebase")
_src = {SRC}
print("merged src:", _src, os.listdir(_src))
if os.path.exists(os.path.join(_src, "rebase_info.json")):  # already rebased + verified locally: use as-is
    REBASED = _src
else:
    _r = subprocess.run(["python", "-m", "brickexp.rebase_qwen35", _src, "/kaggle/working/rebased"], capture_output=True, text=True)
    print(_r.stdout[-2000:], _r.stderr[-3000:])
    assert _r.returncode == 0, "rebase failed"
    REBASED = "/kaggle/working/rebased"
print(REBASED, json.load(open(os.path.join(REBASED, "config.json")))["architectures"])""")
    MODEL_EXPR = "__REBASED__"
else:
    MODEL_EXPR = A.model

if A.train_args or A.merged_glob or A.model != "Qwen/Qwen3.5-2B":
    A.fallback_models = []  # trained / reference models must never silently fall back to another model

code(f"""# ---- config ----
MODEL = {'REBASED' if MODEL_EXPR == '__REBASED__' else repr(MODEL_EXPR)}      # fallback if unsupported: "Qwen/Qwen3.5-0.8B"
BACKEND = "{A.backend}"               # "vllm" or "hf" (transformers batched generate)
PER_BUCKET = {A.per_bucket}                 # prompts per size bucket (small/medium/large)
MAX_BUDGET = {A.max_budget}                # retries; one run gives success@k for all k <= MAX_BUDGET (use 50 for the full curve)
HISTORY_WINDOW = 2             # failed attempts kept in context
CHECKER = "{A.checker}"       # "auto" = Gurobi if licensed (~10-30 s/check on CPU!), else connectivity
TEMPERATURE = 0.6
FALLBACK_MODELS = {A.fallback_models!r}  # tried in order if vLLM can't run MODEL on T4
TIME_BUDGET_H = {A.time_budget_h}   # stop starting new retry rounds after this (partial results still saved)
OUT = f"/kaggle/working/runs/{A.tag}_b{{MAX_BUDGET}}_{{CHECKER}}.jsonl"
""")

code(f"""from brickexp.data import get_eval_set
recs = get_eval_set({A.eval_set})  # fixed, seeded subset shipped in data/splits
recs = [r for b in ("small", "medium", "large") for r in [x for x in recs if x["bucket"] == b][:PER_BUCKET]]
print(len(recs), "prompts")""")

code(f"""import time, torch
from brickexp.backends import make_backend
stage("model load")
ALLOW_HF = {A.allow_hf}
t = time.time()
kw = dict(model=MODEL, temperature=TEMPERATURE, max_tokens={A.max_tokens})
be = None
if BACKEND == "vllm":
    for cand in [MODEL] + FALLBACK_MODELS:
        try:
            extra = dict(limit_mm_per_prompt={{"image": 0, "video": 0}}) if ("Qwen3.5" in cand or "/kaggle/working" in cand or os.path.exists(os.path.join(cand, "preprocessor_config.json"))) else {{}}
            be = make_backend("vllm", **dict(kw, model=cand), dtype="half", tensor_parallel_size=1,
                              max_model_len=16384, enforce_eager={A.enforce_eager},
                              gpu_memory_utilization=0.75, max_num_seqs=64, **extra)  # GDN state lives outside KV budget
            MODEL = cand
            break
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"!!! vLLM failed for {{cand}}: {{repr(e)[:200]}}")
            import gc, torch; gc.collect(); torch.cuda.empty_cache()
    if be is None and not ALLOW_HF:
        raise RuntimeError("vLLM failed for all candidate models")
    if be is None:
        BACKEND = "hf"
print("USING MODEL:", MODEL)
if BACKEND == "hf":
    be = make_backend("hf", **kw, batch_size={A.hf_batch}, adapter_base="{A.hf_adapter_base}" or None)
print(f"model load {{time.time()-t:.0f}}s backend={{BACKEND}}")
stage("probe")
from brickexp import prompts
t = time.time()
probe = be.generate([prompts.initial_messages(r["caption"]) for r in recs[:8]], [dict(r, attempt=0) for r in recs[:8]])
dt = time.time() - t
print(f"probe: 8 prompts in {{dt:.1f}}s; out chars", [len(o) for o in probe])
print(probe[0][:400])""")

code("""import json
from brickexp.evaluate import run_eval, success_curve, failure_breakdown
ug = {"auto": "auto", "gurobi": True, "connectivity": False}[CHECKER]
t = time.time()
stage("eval")
states = run_eval(recs, be, max_budget=MAX_BUDGET, use_gurobi=ug, history_window=HISTORY_WINDOW, out_path=OUT,
                  time_budget_s=TIME_BUDGET_H * 3600)
stage("eval done")
print(f"total {time.time()-t:.0f}s  -> {OUT}")
print(json.dumps(success_curve(states, budgets=(0, 1, 10, 50)), indent=1))
print("failures:", failure_breakdown(states))""")

code("""import pandas as pd
df = pd.DataFrame([{"bucket": s["bucket"], "n_ref": s["n_bricks_ref"], "solved_at": s["solved_at"],
                    "attempts": len(s["attempts"]), "gen_s": round(s["gen_seconds"], 2),
                    "check_s": round(s["check_seconds"], 2), "s_per_attempt": round(s["seconds"] / len(s["attempts"]), 2)}
                   for s in states])
print(df.to_string()); print(df.groupby("bucket")[["gen_s", "check_s", "s_per_attempt"]].mean())""")

code("""# Gurobi cost per check on this CPU (only if licensed), measured on model outputs that passed connectivity
if checker.gurobi_available():
    cands = [a["text"] for s in states for a in s["attempts"] if a["ok"]][:3] or [r["bricks"] for r in recs[:3]]
    for txt in cands:
        t = time.time(); r = checker.check(txt, use_gurobi=True); print(r.n_bricks, r.failure, r.method, f"{time.time()-t:.1f}s")""")

if A.train_args or A.merged_glob:
    code("""!rm -rf /kaggle/working/rebased""")
if A.train_args:
    code(f"""# keep LoRA adapter + logs as output; drop the merged full model (too large to download)
!rm -rf /kaggle/working/{A.tag}/merged
!du -sh /kaggle/working/*""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
p = Path(A.out) if A.out else Path(__file__).resolve().parent.parent / "notebooks" / "kaggle_c0_budget_curve.ipynb"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(p)
