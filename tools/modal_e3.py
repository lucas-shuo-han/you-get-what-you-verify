"""E3: re-run groups A/B/C (V1 recipe: Qwen2.5-1.5B LoRA r16, 1 epoch) with full trajectory tracking, on Modal.

One A10G container per (group, seed). Each job trains with milestones at 0.5 and 1.0 epoch (validation loss + kept
adapter snapshot at exactly those steps, step-0 validation loss, events in trajectory.jsonl), then for every
milestone merges the snapshot and runs k=0 generation on the fixed 90 prompts and the 28 new-category prompts
(vLLM, temperature 0.6, seed 0, connectivity checker after generation). The "base" job evaluates the untrained
model once (step-0 generation, shared by all groups). Resumable: finished stages are skipped.

  modal run --detach tools/modal_e3.py                       # base + A/B/C seed 0
  modal run --detach tools/modal_e3.py --jobs "A:1 B:1 C:1"  # more seeds
  modal volume get rlthelego-e3 / results/e3_modal           # download
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "peft", "datasets", "networkx", "bitsandbytes", "matplotlib")
    .run_commands("pip uninstall -y -q torchao || true")
    .add_local_dir(ROOT / "brickexp", "/repo/brickexp", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "tools", "/repo/tools", ignore=["**/__pycache__", "*.ipynb"])
    .add_local_dir(ROOT / "data" / "mix", "/repo/data/mix")
    .add_local_dir(ROOT / "vendor" / "BrickGPT" / "src" / "brickgpt", "/repo/vendor/BrickGPT/src/brickgpt",
                   ignore=["**/__pycache__"])
)
vol = modal.Volume.from_name("rlthelego-e3", create_if_missing=True)
hf_cache = modal.Volume.from_name("rlthelego-hf-cache", create_if_missing=True)
app = modal.App("rlthelego-e3", image=image)
OUT = "/out"
DATA = {"A": "data/mix/A.jsonl", "B": "data/mix/B.jsonl", "C": "data/mix/C.jsonl"}


def sh(cmd, log):
    import subprocess
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        rc = subprocess.run(cmd, shell=True, cwd="/repo", stdout=f, stderr=subprocess.STDOUT, executable="/bin/bash",
                            env={**__import__("os").environ, "PYTHONPATH": "/repo:/repo/vendor/BrickGPT/src",
                                 "HF_HOME": "/hf"}).returncode
    if rc:
        raise RuntimeError(f"rc={rc}: {cmd}\n" + Path(log).read_text()[-2500:])


def evaluate(model, d: Path, tag: str, log):
    ev = "python -m brickexp.eval_2x2 run --backend vllm --max_budget 0 --checker connectivity"
    for suite, extra in (("90", ""), ("newcat", " --eval_jsonl data/mix/newcat_test_eval.jsonl")):
        if not (d / "runs" / f"summary_{tag}_{suite}.json").exists():
            sh(f"{ev} --out_dir {d}/runs --time_budget_h 2 --model {model} --tag {tag}_{suite}{extra}", log)
            vol.commit()


@app.function(gpu="A10G", volumes={OUT: vol, "/hf": hf_cache}, timeout=10 * 3600)
def run_job(group: str, seed: int) -> str:
    import json, math, shutil
    if group == "base":
        d = Path(OUT) / "base"; d.mkdir(parents=True, exist_ok=True)
        evaluate(BASE, d, "base_step0", d / "run.log")
        return "base done " + json.dumps({s: json.load(open(d / f"runs/summary_base_step0_{s}.json"))["solved_k0"]
                                          for s in ("90", "newcat")})
    d = Path(OUT) / f"{group}_s{seed}"; d.mkdir(parents=True, exist_ok=True)
    log = d / "run.log"
    from transformers import AutoTokenizer
    import sys; sys.path.insert(0, "/repo")
    from brickexp.train_sft import tokenize
    tok = AutoTokenizer.from_pretrained(BASE, cache_dir="/hf")
    msgs = [json.loads(l)["messages"] for l in open("/repo/" + DATA[group], encoding="utf-8") if l.strip()]
    n = sum(len(tokenize(tok, m, 10 ** 9)["input_ids"]) <= 3072 for m in msgs)
    total = math.ceil(n / 16)
    milestones = [math.ceil(total / 2), total]
    train = (f"python -m brickexp.train_sft --model {BASE} --data_jsonl {DATA[group]} --out {d}/train --seed {seed} "
             f"--bs 2 --grad_accum 8 --max_len 3072 --rank 16 --lr 2e-4 --max_steps {total} --save_steps 40 "
             f"--no_merge --eval_jsonl data/mix/heldout200.jsonl --eval_steps 40 --snap_steps 80")
    for step in milestones:
        snap = d / "train" / "traj" / f"step_{step}"
        if not (snap / "adapter_model.safetensors").exists():
            sh(train + f" --stop_at {step}", log)
            vol.commit()
        tag = f"{group}_s{seed}_step{step}"
        if not (d / "runs" / f"summary_{tag}_newcat.json").exists():
            merged = Path("/tmp/merged"); shutil.rmtree(merged, ignore_errors=True)
            sh(f"python -c \"import torch;from peft import PeftModel;from transformers import AutoModelForCausalLM as M,"
               f"AutoTokenizer as T;b=M.from_pretrained('{BASE}',dtype=torch.float16,device_map={{'':0}});"
               f"PeftModel.from_pretrained(b,'{snap}').merge_and_unload().save_pretrained('{merged}');"
               f"T.from_pretrained('{BASE}').save_pretrained('{merged}')\"", log)
            evaluate(str(merged), d, tag, log)
            shutil.rmtree(merged, ignore_errors=True)
    sh(f"python tools/plot_trajectory.py {d}/train --milestones {','.join(map(str, milestones))}", log)
    vol.commit()
    chk = json.load(open(d / "train" / "trajectory_check.json"))
    k0 = {s: json.load(open(d / f"runs/summary_{group}_s{seed}_step{milestones[-1]}_{s}.json"))["solved_k0"]
          for s in ("90", "newcat")}
    return f"{group}_s{seed} done n={n} steps={total} final k0 valid={k0} problems={chk['problems']}"


@app.local_entrypoint()
def main(jobs: str = "base:0 A:0 B:0 C:0"):
    args = [(j.split(":")[0], int(j.split(":")[1])) for j in jobs.split()]
    for res in run_job.starmap(args, return_exceptions=True):
        print(res if not isinstance(res, Exception) else f"ERROR {res!r}", flush=True)
