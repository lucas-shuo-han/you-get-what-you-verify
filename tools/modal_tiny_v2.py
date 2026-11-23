"""Run the DIRECTION §4d v2 tiny 4-arm experiment on Modal (one T4 container per job, in parallel).

Usage (repo root):
  modal run tools/modal_tiny_v2.py                    # all 12 jobs (RF/RFs/R/S x seeds 0,1,2)
  modal run tools/modal_tiny_v2.py --jobs RF:0        # smoke: one job
  modal volume get rlthelego-tiny-v2 / results/tiny_4arm/modal_v2   # download outputs

Each job: train (v2 defaults: --loss_mask bad, padded R) + unmasked eval k<=10 with raw tokens, then an output check
(420 records, raw present, eval_curve present, post_filter per-brick JSON written). Resumable: a job whose output
passes the check is skipped; a job with model.pt but no eval runs --eval_only.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "networkx", "gurobipy")
    .add_local_dir(ROOT / "brickexp", "/repo/brickexp", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "data" / "tiny" / "n4000_s0", "/repo/data/tiny/n4000_s0")
    .add_local_dir(ROOT / "vendor" / "BrickGPT" / "src" / "brickgpt", "/repo/vendor/BrickGPT/src/brickgpt",
                   ignore=["**/__pycache__"])
)
vol = modal.Volume.from_name("rlthelego-tiny-v2", create_if_missing=True)
app = modal.App("rlthelego-tiny-v2", image=image)
OUT = "/out"


def check(d: Path) -> str:
    import json, subprocess, sys
    recs = [json.loads(l) for l in open(d / "eval.jsonl")]
    assert len(recs) == 420, f"{len(recs)} records"
    assert all(r["attempts"] and all("raw" in a for a in r["attempts"]) for r in recs), "missing attempts/raw"
    s = json.load(open(d / "train_log.json"))["summary"]
    assert any(k.startswith("eval_curve") for k in s), "no eval_curve"
    out = d / "per_brick_postfilter_k0.json"
    subprocess.run([sys.executable, "-m", "brickexp.tiny.per_brick_metrics", str(d / "eval.jsonl"), "--post_filter",
                    "--k", "0", "--boot", "10000", "--out", str(out)], check=True, cwd="/repo",
                   capture_output=True)
    pb = json.load(open(out))[0]
    assert pb["n_prompts"] == 420
    return f"CHECK_OK {d.name} attempts={sum(len(r['attempts']) for r in recs)} curve={s.get('eval_curve')} " \
           f"floating={pb['floating_frac']:.4f}"


@app.function(gpu="T4", volumes={OUT: vol}, timeout=3 * 3600)
def run_job(arm: str, seed: int) -> str:
    import subprocess, sys
    d = Path(OUT) / f"{arm}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        return "SKIP " + check(d)
    except Exception:
        pass
    mode = ["--eval_only"] if (d / "model.pt").exists() and (d / "train_log.json").exists() else []
    with open(d / "run.log", "a") as log:
        rc = subprocess.run([sys.executable, "-m", "brickexp.tiny.train", "--data", "data/tiny/n4000_s0", "--arm", arm,
                             "--seed", str(seed), "--out", str(d), *mode], cwd="/repo", stdout=log,
                            stderr=subprocess.STDOUT).returncode
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"FAIL {arm}_s{seed} rc={rc}: " + (d / "run.log").read_text()[-1500:])
    msg = check(d)
    vol.commit()
    return msg


@app.local_entrypoint()
def main(jobs: str = "RF:0 RFs:0 R:0 S:0 RF:1 RFs:1 R:1 S:1 RF:2 RFs:2 R:2 S:2"):
    args = [(j.split(":")[0], int(j.split(":")[1])) for j in jobs.split()]
    for res in run_job.starmap(args, return_exceptions=True):
        print(res if not isinstance(res, Exception) else f"ERROR {res!r}", flush=True)
