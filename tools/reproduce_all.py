"""Regenerate every paper figure and the statistics table from cached results (no GPU, no Gurobi needed).

python tools/reproduce_all.py
"""
import shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for script in ("paper/make_figs.py", "results/paper/curves_stats.py"):
    print(">>", script, flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], check=True, cwd=ROOT)
shutil.copy(ROOT / "results/paper/train_curves.png", ROOT / "paper/figs/train_curves.png")
print("figures in paper/figs/, table in results/paper/stats_table.md")
