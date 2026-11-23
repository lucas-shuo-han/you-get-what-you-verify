"""Re-judge downloaded trajectories with Gurobi locally (license never leaves this machine).

For each attempt that passed the online checker (connectivity), run Gurobi on that structure.
Writes <in>.gurobi.jsonl where each state gets:
  solved_at_gurobi : first attempt index whose structure passes Gurobi AND was accepted online
                     (k=0 strictly valid; k>0 = "connectivity feedback + Gurobi final verdict")
Only accepted attempts need Gurobi (rejected ones already failed a necessary condition).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .checker import check, gurobi_available
from .data import load_jsonl


def rejudge_file(path: str, time_limit: float = 120) -> Path:
    if not gurobi_available():
        raise SystemExit("Gurobi not available locally")
    states = load_jsonl(path)
    cache, t0 = {}, time.time()
    for s in states:
        s["solved_at_gurobi"] = None
        for a in s["attempts"]:
            if not a["ok"]:
                continue
            key = a["text"]
            if key not in cache:
                cache[key] = check(key, use_gurobi=True, gurobi_kwargs={"time_limit": time_limit}).to_dict()
            a["gurobi"] = cache[key]
            if cache[key]["ok"] and s["solved_at_gurobi"] is None:
                s["solved_at_gurobi"] = a["attempt"]
    out = Path(str(path).replace(".jsonl", ".gurobi.jsonl"))
    with open(out, "w", encoding="utf-8") as f:
        for s in states:
            s["solved_at_connectivity"] = s["solved_at"]
            s["solved_at"] = s["solved_at_gurobi"]  # analysis uses solved_at
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"{path}: {len(cache)} gurobi checks in {time.time() - t0:.0f}s -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--time_limit", type=float, default=120)
    a = ap.parse_args()
    for f in a.files:
        if not f.endswith(".gurobi.jsonl"):
            rejudge_file(f, a.time_limit)
