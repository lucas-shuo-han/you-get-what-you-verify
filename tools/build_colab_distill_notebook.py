"""Colab version of the distill-SFT + 2x2 notebook (free T4, fp16), derived from tools/build_distill_notebook.py
so the two never drift. Differences vs Kaggle:
  - outputs live on Google Drive (OUTD = /content/drive/MyDrive/brickexp_distill): adapter, logs, eval jsonl,
    summaries survive a disconnect; each stage is skipped on re-run if its output already exists
    (train: sft/train_log.json + sft/merged; eval: runs/summary_<tag>.json). An eval interrupted mid-way restarts
    that eval from scratch (<= time budget).
  - bundle: brickexp_code.zip from Drive (OUTD/brickexp_code.zip) or uploaded to /content.
  - checker = connectivity, same as the C1 Kaggle eval (its Gurobi self-check failed -> connectivity fallback);
    stability is re-judged locally with Gurobi afterwards (brickexp/rejudge.py), as for C1. No secrets needed.
python tools/build_colab_distill_notebook.py [--limit 0]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--time_budget_h", type=float, default=3.0)
ap.add_argument("--out", default=str(ROOT / "notebooks/colab_distill_sft.ipynb"))
a = ap.parse_args()
tmp = ROOT / "dist/_kaggle_base.ipynb"
subprocess.run([sys.executable, str(ROOT / "tools/build_distill_notebook.py"), "--checker", "connectivity",
                "--limit", str(a.limit), "--time_budget_h", str(a.time_budget_h), "--out", str(tmp)], check=True)
nb = json.loads(tmp.read_text(encoding="utf-8"))
out = []
for c in nb["cells"]:
    s = "".join(c["source"])
    if c["cell_type"] == "markdown":
        s = ("# Distill SFT (Colab, free T4) + 2x2 eval\nRuntime → GPU (T4). Put `brickexp_code.zip` in "
             "`MyDrive/brickexp_distill/` (or upload it to /content). Run all; after a disconnect just Run all again — "
             "finished stages are skipped. Checker = connectivity (same as C1); Gurobi re-judge happens locally.")
    elif 'OUTD = "/kaggle/working"' in s:
        s = s.replace('OUTD = "/kaggle/working" if os.path.exists("/kaggle") else "/content/work"',
                      'from google.colab import drive\ndrive.mount("/content/drive")\n'
                      'OUTD = "/content/drive/MyDrive/brickexp_distill"')
        s = s.replace('pip install -q vllm peft gurobipy datasets networkx bitsandbytes',
                      'pip install -q vllm peft gurobipy datasets networkx bitsandbytes')
    elif "brickexp_code.zip" in s and "WORK =" in s:
        s = s.replace('WORK = OUTD + "/RLtheLEGO"', 'WORK = "/content/RLtheLEGO"  # code on local disk, outputs on Drive')
        s = s.replace('glob.glob("/kaggle/input/**/brickexp_code.zip", recursive=True)',
                      'glob.glob(OUTD + "/brickexp_code.zip")')
    elif "kaggle_secrets" in s:
        s = ('# no secrets on Colab: checker = connectivity (as in the C1 eval); stability re-judged locally later\n'
             'from brickexp import checker\nprint("Gurobi available here:", checker.gurobi_available())\n'
             'n = sum(1 for _ in open("data/distill/sft_distill.jsonl")); print("SFT examples:", n); assert n > 0')
    elif "brickexp.train_sft" in s:
        s = ("DONE_TRAIN = os.path.exists(f\"{OUTD}/sft/train_log.json\") and (os.path.exists(f\"{OUTD}/sft/merged/config.json\") or os.path.exists(f\"{OUTD}/runs/summary_sft.json\"))\n"
             "if DONE_TRAIN:\n    print('train: already done on Drive, skipping')\nelse:\n"
             + "\n".join("    " + l for l in s.splitlines()))
    elif "eval_2x2 run" in s:
        tag = s.split("--tag ")[1].split('"')[0].split()[0]
        s = (f"if os.path.exists(f\"{{OUTD}}/runs/summary_{tag}.json\"):\n    print('eval {tag}: done, skipping')\nelse:\n    " + s)
    elif "eval_2x2 table" in s:
        s = s.replace(' {OUTD}/RLtheLEGO', '')  # keep merged on Drive until the user deletes it? no: remove merged, keep adapter
    c["source"] = s
    out.append(c)
nb["cells"] = out
nb["metadata"]["accelerator"] = "GPU"
Path(a.out).write_text(json.dumps(nb, indent=1), encoding="utf-8")
tmp.unlink()
print(a.out)
