"""Kaggle notebook for from-scratch tiny-transformer runs (DIRECTION v2). Observable: hello.txt, run.log,
stage_*.txt, per-job logs, error.txt; any failure stops the run.

--jobs "ARM:SEED:EXTRA_ARGS;ARM:SEED:EXTRA_ARGS"   e.g. "RF:0:--smoke 20;S:0:--smoke 20"
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)
ap.add_argument("--data", default="data/tiny/n4000_s0")
ap.add_argument("--out", required=True)
A = ap.parse_args()

cells = []
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

code("""import os, sys, time, traceback, datetime, subprocess, glob, zipfile
OUTD = "/kaggle/working"; os.makedirs(OUTD, exist_ok=True)
open(f"{OUTD}/hello.txt", "w").write(f"hello {datetime.datetime.utcnow().isoformat()}\\n")
_RUNLOG = open(f"{OUTD}/run.log", "a", buffering=1)
class _Tee:
    def __init__(self, s): self.s = s
    def write(self, x): self.s.write(x); _RUNLOG.write(x)
    def flush(self): self.s.flush(); _RUNLOG.flush()
    def __getattr__(self, k): return getattr(self.s, k)
sys.stdout, sys.stderr = _Tee(sys.stdout), _Tee(sys.stderr)
def _on_error(shell, etype, evalue, tb, tb_offset=None):
    msg = "".join(traceback.format_exception(etype, evalue, tb))
    open(f"{OUTD}/error.txt", "a").write(msg); _RUNLOG.write(msg); _RUNLOG.flush()
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset); os._exit(1)
get_ipython().set_custom_exc((BaseException,), _on_error)
T0 = time.time()
def stage(name):
    print(f"=== [{(time.time()-T0)/60:6.1f} min] {name}", flush=True)
    open(f"{OUTD}/stage_{int(time.time()-T0):06d}_{name.replace(' ', '_').replace(':', '-')}.txt", "w").write(time.ctime())
stage("start")
WORK = "/kaggle/working/RLtheLEGO"
zips = glob.glob("/kaggle/input/**/brickexp_code.zip", recursive=True)
assert zips, "code dataset not attached"
zipfile.ZipFile(zips[0]).extractall(WORK)
os.chdir(WORK); sys.path.insert(0, WORK)
r = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; python -c 'import torch;print(torch.__version__)'",
                   shell=True, capture_output=True, text=True); print(r.stdout, r.stderr[-300:])
r = subprocess.run("pip install -q networkx 2>&1 | tail -2", shell=True, capture_output=True, text=True); print(r.stdout)
r = subprocess.run("python -m pytest -q tests -k 'tiny or floating or collision' 2>&1 | tail -3", shell=True,
                   capture_output=True, text=True); print(r.stdout)""")

code(f"""JOBS = {json.dumps([j.split(":", 2) for j in A.jobs.split(";") if j])}
DATA = "{A.data}"
def run_logged(cmd, tag):
    stage(tag)
    p = subprocess.run(f"set -o pipefail; ( {{cmd}} ) 2>&1 | tee /kaggle/working/{{tag}}.log", shell=True,
                       executable="/bin/bash", capture_output=True, text=True)
    print(p.stdout[-3000:])
    if p.returncode != 0:
        open("/kaggle/working/error.txt", "a").write(f"[{{tag}}] rc={{p.returncode}}\\n" + p.stdout[-3000:])
        raise RuntimeError(f"{{tag}} failed rc={{p.returncode}}")
for arm, seed, extra in JOBS:
    out = f"/kaggle/working/runs/{{arm}}_s{{seed}}"
    run_logged(f"python -m brickexp.tiny.train --data {{DATA}} --arm {{arm}} --seed {{seed}} --out {{out}} {{extra}}",
               f"job_{{arm}}_s{{seed}}")
stage("done")""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(A.out, "w"), indent=1)
print(A.out)
