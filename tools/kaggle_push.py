"""Fully automatic Kaggle run of the C0 notebook.

1. bundle code -> private dataset <user>/brickexp-code (create or new version)
2. build notebook + kernel-metadata.json (private, GPU T4x2, internet) -> kaggle kernels push
3. poll status, then download output to results/c0/

Auth uses ~/.kaggle/kaggle.json via the kaggle library; credentials are never read or printed here.
Usage: python tools/kaggle_push.py [--per_bucket 10 --max_budget 10 --checker connectivity]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "dist" / "kaggle"
DS_SLUG = "brickexp-code"
KN_SLUG = "brickexp-c0-budget-curve"  # overridden by --kernel


def sh(*args, check=True):
    _token_env()
    print("$ kaggle", *args, flush=True)
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip()[-2000:], flush=True)
    if check and r.returncode != 0:
        sys.exit(f"kaggle {args[0]} {args[1]} failed")
    return r.returncode, out


def _token_env():
    """Support both legacy kaggle.json and a new-style raw API token saved in that file (never printed)."""
    import os
    f = Path.home() / ".kaggle" / "kaggle.json"
    if f.exists() and not (Path.home() / ".kaggle" / "access_token").exists():
        txt = f.read_text().strip()
        if txt and not txt.startswith("{"):
            os.environ["KAGGLE_API_TOKEN"] = txt


def username():
    _token_env()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    u = api.get_config_value("username")
    if not u:  # new-style tokens: ask the API who we are
        try:
            u = api.get_current_user().user_name  # noqa
        except Exception:
            pass
    if not u:
        sys.exit("could not determine Kaggle username; pass --user")
    return u


def push_dataset(user):
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_kaggle_bundle.py")], check=True)
    d = STAGE / "dataset"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy(ROOT / "dist" / "brickexp_code.zip", d)
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": DS_SLUG, "id": f"{user}/{DS_SLUG}", "licenses": [{"name": "other"}]}))
    rc, out = sh("datasets", "status", f"{user}/{DS_SLUG}", check=False)
    if rc == 0 and "ready" in out.lower():
        sh("datasets", "version", "-p", str(d), "-m", f"update {time.strftime('%Y-%m-%d %H:%M')}")
    else:
        sh("datasets", "create", "-p", str(d))  # private by default
    for _ in range(60):  # wait until processed
        time.sleep(10)
        _, out = sh("datasets", "status", f"{user}/{DS_SLUG}", check=False)
        if "ready" in out.lower():
            return
    sys.exit("dataset not ready after 10 min")


def push_kernel(user, a):
    k = STAGE / "kernel"
    shutil.rmtree(k, ignore_errors=True)
    k.mkdir(parents=True)
    nb = k / "c0.ipynb"
    if a.tiny_jobs:
        subprocess.run([sys.executable, str(ROOT / "tools" / "build_tiny_notebook.py"), "--jobs", a.tiny_jobs,
                        "--data", a.tiny_data, "--out", str(nb)], check=True)
    else:
      subprocess.run([sys.executable, str(ROOT / "tools" / "build_notebook.py"), "--per_bucket", str(a.per_bucket),
                    "--max_budget", str(a.max_budget), "--checker", a.checker, "--model", a.model,
                    "--time_budget_h", str(a.time_budget_h), "--max_tokens", str(a.max_tokens),
                    "--vllm_spec", a.vllm_spec, "--backend", a.backend, "--hf_batch", str(a.hf_batch),
                    *(["--enforce_eager"] if a.enforce_eager else []),
                    *(["--allow_hf"] if a.allow_hf else []),
                    *(["--hf_adapter_base", a.hf_adapter_base] if a.hf_adapter_base else []),
                    "--tag", a.tag, "--eval_set", str(a.eval_set), "--train_args", a.train_args,
                    *(["--merged_glob", "/kaggle/input/**/*.safetensors"] if a.merged_dataset else []),
                    "--out", str(nb)], check=True)
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KN_SLUG}", "title": KN_SLUG, "code_file": nb.name, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/{DS_SLUG}"] + ([a.merged_dataset] if a.merged_dataset else []), "competition_sources": [], "kernel_sources": [],
        "machine_shape": "NvidiaTeslaT4"}, indent=1))
    sh("kernels", "push", "-p", str(k), "--accelerator", "NvidiaTeslaT4", "-t", str(int(a.kernel_timeout_h * 3600)))


def wait_and_fetch(user, poll):
    t0 = time.time()
    while True:
        time.sleep(poll)
        _, out = sh("kernels", "status", f"{user}/{KN_SLUG}", check=False)
        low = out.lower()
        print(f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        if any(s in low for s in ("complete", "error", "cancel")):
            break
    dst = ROOT / "results" / TAG
    dst.mkdir(parents=True, exist_ok=True)
    sh("kernels", "output", f"{user}/{KN_SLUG}", "-p", str(dst), "-o", check=False)
    return "complete" in low and "error" not in low


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_bucket", type=int, default=10)
    ap.add_argument("--max_budget", type=int, default=10)
    ap.add_argument("--checker", default="connectivity")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--skip_dataset", action="store_true")
    ap.add_argument("--fetch_only", action="store_true")
    ap.add_argument("--user", default=None)
    ap.add_argument("--tag", default="c0")
    ap.add_argument("--kernel", default=None, help="kernel slug (default brickexp-<tag>)")
    ap.add_argument("--eval_set", type=int, default=10)
    ap.add_argument("--train_args", default="")
    ap.add_argument("--merged_dataset", default="", help="owner/slug of a Kaggle dataset holding a merged model dir")
    ap.add_argument("--time_budget_h", type=float, default=8.0)
    ap.add_argument("--kernel_timeout_h", type=float, default=11.5)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--vllm_spec", default="vllm")
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--hf_batch", type=int, default=32)
    ap.add_argument("--allow_hf", action="store_true")
    ap.add_argument("--hf_adapter_base", default="")
    ap.add_argument("--tiny_jobs", default="", help="ARM:SEED:ARGS;... -> from-scratch tiny-transformer notebook")
    ap.add_argument("--tiny_data", default="data/tiny/n4000_s0")
    ap.add_argument("--no_wait", action="store_true")
    a = ap.parse_args()
    TAG = a.tag
    KN_SLUG = a.kernel or ("brickexp-c0-budget-curve" if a.tag == "c0" else f"brickexp-{a.tag.replace('_', '-')}")
    _token_env()
    user = a.user or username()
    if not a.fetch_only:
        if not a.skip_dataset:
            push_dataset(user)
        push_kernel(user, a)
    if a.no_wait:
        sys.exit(0)
    ok = wait_and_fetch(user, a.poll)
    print("DONE ok=" + str(ok))
