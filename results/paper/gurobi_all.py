"""Gurobi re-check (attempt 0 = k=0) of every mix run used in the paper. Writes results/paper/gurobi_all.json."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from brickexp.checker import check
RUNS = {"base": "mix_B/runs/eval_base", "A_s0": "mix_A/runs/eval_A", "B_s0": "mix_B/runs/eval_B",
        "B_s1": "final_small/b1x1/runs/eval_B_s1_x1", "C_s0": "final_small/c0x1/runs/eval_C_s0_x1"}
extra = Path(ROOT / "results/paper/extra_runs.json")  # e.g. A_s1 once downloaded
if extra.exists():
    RUNS.update(json.load(open(extra)))
out = {}
for name, stem in RUNS.items():
    for suite in ("90", "newcat"):
        p = ROOT / "results" / f"{stem}_{suite}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        ok_ids, fails = [], {}
        for r in rows:
            for _ in range(3):  # checker_error = transient Gurobi timeout/error under CPU load -> retry (DEBUG_LOG 11-23)
                res = check(r["attempts"][0].get("raw") or "", use_gurobi=True)
                if res.failure != "checker_error":
                    break
            assert res.method != "connectivity"
            if res.ok:
                ok_ids.append(r["structure_id"])
            else:
                fails[res.failure] = fails.get(res.failure, 0) + 1
        out[f"{name}_{suite}"] = {"n": len(rows), "gurobi_ok": len(ok_ids), "fails": fails, "ok_ids": ok_ids,
                                  "connectivity_ok": sum(r["solved_at"] == 0 for r in rows), "file": str(p.relative_to(ROOT))}
        print(name, suite, len(ok_ids), "/", len(rows), flush=True)
json.dump(out, open(ROOT / "results/paper/gurobi_all.json", "w"), indent=1)
