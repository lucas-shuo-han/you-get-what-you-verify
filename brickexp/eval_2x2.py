"""2x2 eval for the distillation experiment: {base, distill-SFT} x {one-shot k=0, feedback <=3 rounds}.

One run with max_budget=3 gives both columns (success@0 and success@3), reusing evaluate.run_eval + checker.
  python -m brickexp.eval_2x2 run --model Qwen/Qwen2.5-1.5B-Instruct --tag base --out_dir runs/distill
  python -m brickexp.eval_2x2 run --model runs/distill/sft/merged --tag sft --out_dir runs/distill
  python -m brickexp.eval_2x2 run --model Qwen/Qwen2.5-1.5B-Instruct --adapter runs/distill/sft/adapter --backend hf ...
  python -m brickexp.eval_2x2 table runs/distill/summary_base.json runs/distill/summary_sft.json
All heavy imports (torch / vllm / gurobipy) are lazy.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def cmd_run(a):
    from .backends import make_backend
    from .data import get_eval_set
    from .evaluate import run_eval, success_curve, failure_breakdown
    if a.eval_jsonl:  # custom prompt set (e.g. new categories): rows need structure_id/caption/bucket
        recs = [json.loads(l) for l in open(a.eval_jsonl, encoding="utf-8") if l.strip()]
        for r in recs:
            r.setdefault("n_bricks", 0)
    else:
        recs = get_eval_set(a.per_bucket)  # 30/bucket = the same 90 prompts as Ref / C1 / DeepSeek pilot
    if a.limit:
        recs = [r for b in ("small", "medium", "large") for r in [x for x in recs if x["bucket"] == b][:a.limit]]
    kw = dict(temperature=a.temperature, max_tokens=a.max_tokens)
    if a.backend == "vllm":
        be = make_backend("vllm", model=a.model, dtype="half", max_model_len=a.max_model_len,
                          gpu_memory_utilization=a.gpu_mem, **kw)
    elif a.backend == "hf":
        if a.adapter:  # base + LoRA adapter (merged in memory)
            be = make_backend("hf", model=a.adapter, adapter_base=a.model, batch_size=a.hf_batch, **kw)
        else:
            be = make_backend("hf", model=a.model, batch_size=a.hf_batch, **kw)
    else:
        be = make_backend("replay")
    ug = {"auto": "auto", "gurobi": True, "connectivity": False}[a.checker]
    out_dir = Path(a.out_dir)
    t0 = time.time()
    states = run_eval(recs, be, max_budget=a.max_budget, use_gurobi=ug, history_window=a.history_window,
                      out_path=out_dir / f"eval_{a.tag}.jsonl", time_budget_s=a.time_budget_h * 3600)
    budgets = tuple(sorted({0, a.max_budget}))
    curve = success_curve(states, budgets=budgets)
    methods = sorted({x.get("method") or "" for s in states for x in s["attempts"]} - {""})
    summ = {"tag": a.tag, "model": a.model, "adapter": a.adapter, "n": len(states), "checker": a.checker,
            "checker_methods_seen": methods, "max_budget": a.max_budget,
            "max_budget_reached": min(s["max_budget"] for s in states),
            "solved_k0": sum(s["solved_at"] == 0 for s in states),
            "solved_kmax": sum(s["solved_at"] is not None for s in states),
            "curve": curve, "failures": failure_breakdown(states), "seconds": time.time() - t0}
    (out_dir / f"summary_{a.tag}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def cmd_table(a):
    rows = [json.loads(Path(p).read_text()) for p in a.summaries]
    kmax = rows[0]["max_budget"]
    lines = [f"| model | one-shot k=0 | feedback <= {kmax} rounds | checker |", "|---|---|---|---|"]
    for r in rows:
        note = "" if r["max_budget_reached"] >= r["max_budget"] else f" (only k<={r['max_budget_reached']} reached)"
        lines.append(f"| {r['tag']} ({r['model']}) | {r['solved_k0']}/{r['n']} ({r['solved_k0'] / r['n']:.0%}) | "
                     f"{r['solved_kmax']}/{r['n']} ({r['solved_kmax'] / r['n']:.0%}){note} | "
                     f"{','.join(r['checker_methods_seen']) or r['checker']} |")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--adapter", default=None, help="hf backend: LoRA adapter dir applied on --model")
    r.add_argument("--tag", required=True)
    r.add_argument("--out_dir", default="runs/distill")
    r.add_argument("--backend", default="vllm", choices=["vllm", "hf", "replay"])
    r.add_argument("--per_bucket", type=int, default=30)
    r.add_argument("--eval_jsonl", default=None, help="custom prompt rows (structure_id, caption, bucket) instead of the 90")
    r.add_argument("--limit", type=int, default=0, help="first N prompts per bucket (smoke)")
    r.add_argument("--max_budget", type=int, default=3)
    r.add_argument("--history_window", type=int, default=2)
    r.add_argument("--temperature", type=float, default=0.6)
    r.add_argument("--max_tokens", type=int, default=3072)
    r.add_argument("--max_model_len", type=int, default=16384)
    r.add_argument("--gpu_mem", type=float, default=0.8)
    r.add_argument("--hf_batch", type=int, default=16)
    r.add_argument("--checker", default="gurobi", choices=["auto", "gurobi", "connectivity"])
    r.add_argument("--time_budget_h", type=float, default=4.0)
    r.set_defaults(fn=cmd_run)
    t = sp.add_parser("table")
    t.add_argument("summaries", nargs="+")
    t.add_argument("--out", default=None)
    t.set_defaults(fn=cmd_table)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
