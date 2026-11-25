"""Gurobi-free re-check: recount one-shot (k=0) validity of every cached student run with the
connectivity checker (format, bounds, collision, floating, connectivity; no force-balance LP).

These are the "connectivity" counts reported next to the Gurobi counts in the paper
(e.g. pre-registration appendix and the exploratory 300-prompt evaluation). Gurobi counts are
equal or lower because the LP additionally rejects statically unstable structures.

Usage:  python tools/recheck_connectivity.py          (CPU only, no license, ~1 min)
"""
import glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brickexp.checker import check  # noqa: E402

PATTERNS = [
    "results/mix_A/runs/eval_*.jsonl",
    "results/mix_B/runs/eval_*.jsonl",
    "results/final_small/*/runs/eval_*.jsonl",
    "results/paper/as1/runs/eval_*.jsonl",
    "results/paper/cs1/eval_*.jsonl",
    "results/paper/eval300/runs/eval_*.jsonl",
    "results/paper/eval300/s1/runs/eval_*.jsonl",
]

rows_out = []
for pat in PATTERNS:
    for f in sorted(glob.glob(str(ROOT / pat))):
        if f.endswith(".gurobi.jsonl"):
            continue
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        ok = sum(check((r["attempts"][0].get("raw") or ""), use_gurobi=False).ok for r in rows)
        rows_out.append((str(Path(f).relative_to(ROOT)).replace("\\", "/"), ok, len(rows)))

w = max(len(r[0]) for r in rows_out)
print(f"{'run file'.ljust(w)}  connectivity-valid at k=0")
for name, ok, n in rows_out:
    print(f"{name.ljust(w)}  {ok:>3} / {n}")
out = ROOT / "results/connectivity_recheck.json"
out.write_text(json.dumps({n: {"ok": ok, "n": t} for n, ok, t in rows_out}, indent=1))
print(f"wrote {out.relative_to(ROOT)}")
