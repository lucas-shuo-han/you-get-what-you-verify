"""Gurobi re-check of mix A/B/base eval outputs (attempt 0). Writes results/mix_gurobi_recheck.json."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from brickexp.checker import check
out = {}
for f in ["mix_A/runs/eval_A_90", "mix_A/runs/eval_A_newcat", "mix_B/runs/eval_B_90", "mix_B/runs/eval_B_newcat",
          "mix_B/runs/eval_base_90", "mix_B/runs/eval_base_newcat"]:
    rows = [json.loads(l) for l in open(ROOT / "results" / (f + ".jsonl"), encoding="utf-8")]
    ok = 0; fails = {}; ids = []
    for r in rows:
        t = r["attempts"][0].get("raw") or ""
        res = check(t, use_gurobi=True)
        assert res.method != "connectivity"
        if res.ok: ok += 1; ids.append(r["structure_id"])
        else: fails[res.failure] = fails.get(res.failure, 0) + 1
    out[f.split("/")[-1]] = {"n": len(rows), "gurobi_ok": ok, "fails": fails, "ok_ids": ids}
    print(f, ok, "/", len(rows), fails, flush=True)
json.dump(out, open(ROOT / "results/mix_gurobi_recheck.json", "w"), indent=1)
