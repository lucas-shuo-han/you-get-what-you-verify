"""Whole-structure generate -> check -> feedback -> retry loop.

One run with max budget K yields the curve for every k <= K:
success@k = solved within the first k+1 attempts (k = number of retries; k=0 is pure intuition).
No per-brick rejection sampling: the model always emits the full structure.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from . import prompts
from .checker import check, extract_bricks_text

DEFAULT_BUDGETS = (0, 1, 10, 50)


def build_conversation(caption: str, history: list[dict], history_window: int | None) -> list[dict]:
    """history: failed attempts [{'text','message'}]; only the last `history_window` are kept (None = all)."""
    conv = prompts.initial_messages(caption)
    if history_window is not None:
        history = history[-history_window:] if history_window > 0 else []
    for a in history:
        conv.append({"role": "assistant", "content": a["text"]})
        conv.append({"role": "user", "content": prompts.feedback(a["message"])})
    return conv


def run_eval(records: list[dict], backend, max_budget: int = 50, world_dim: int = 20,
             use_gurobi="auto", history_window: int | None = 2, out_path: str | Path | None = None,
             verbose: bool = True, gurobi_kwargs: dict | None = None,
             time_budget_s: float | None = None) -> list[dict]:
    """time_budget_s: stop starting new retry rounds after this wall time; max_budget is then
    recorded as the last completed round so curves never claim unreached k. Saves after every round."""
    t0 = time.time()
    cache: dict[str, dict] = {}  # identical structures are checked once (Gurobi is ~10-30 s/structure)
    states = [{"idx": i, "structure_id": r["structure_id"], "bucket": r["bucket"], "n_bricks_ref": r["n_bricks"],
               "caption": r["caption"], "attempts": [], "solved_at": None,
               "gen_seconds": 0.0, "check_seconds": 0.0}
              for i, r in enumerate(records)]
    active = list(range(len(records)))
    done_rounds = -1

    def save(final_budget):
        for s in states:
            s["seconds"] = s["gen_seconds"] + s["check_seconds"]
            s["max_budget"] = final_budget
            s["history_window"] = history_window
        if out_path:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                for s in states:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

    for attempt in range(max_budget + 1):
        if not active:
            break
        if time_budget_s and time.time() - t0 > time_budget_s:
            print(f"time budget hit before attempt {attempt}", flush=True)
            break
        convs = [build_conversation(states[i]["caption"], states[i]["attempts"], history_window) for i in active]
        metas = [{**records[i], "attempt": attempt} for i in active]
        tg = time.time()
        outs = backend.generate(convs, metas)
        share = (time.time() - tg) / len(active)  # batched: split wall time evenly
        still = []
        for i, raw in zip(active, outs):
            s = states[i]
            s["gen_seconds"] += share
            tc = time.time()
            key = extract_bricks_text(raw)
            if key not in cache:
                cache[key] = check(raw, world_dim=world_dim, use_gurobi=use_gurobi,
                                   gurobi_kwargs=gurobi_kwargs).to_dict()
            res = cache[key]
            s["check_seconds"] += time.time() - tc
            s["attempts"].append({"attempt": attempt, "raw": raw, "text": key or raw, **res})
            if res["ok"]:
                s["solved_at"] = attempt
            else:
                still.append(i)
        active = still
        done_rounds = attempt
        save(done_rounds)
        if verbose:
            n = len(records)
            mean_chars = sum(len(o) for o in outs) / len(outs)
            print(f"[attempt {attempt}] solved {n - len(active)}/{n}  round {time.time() - tg:.1f}s "
                  f"mean_out_chars {mean_chars:.0f}  elapsed {time.time() - t0:.1f}s", flush=True)
    # everything solved early => every k up to max_budget is valid; otherwise only completed rounds
    save(max_budget if not active else done_rounds)
    return states


def success_curve(states: list[dict], budgets=DEFAULT_BUDGETS) -> dict:
    """{bucket|'all': {k: success rate}}; budgets above the run's max_budget are skipped."""
    groups = defaultdict(list)
    for s in states:
        groups[s["bucket"]].append(s)
        groups["all"].append(s)
    out = {}
    for g, ss in groups.items():
        mb = min(s["max_budget"] for s in ss)
        out[g] = {k: sum(s["solved_at"] is not None and s["solved_at"] <= k for s in ss) / len(ss)
                  for k in budgets if k <= mb}
    return out


def failure_breakdown(states: list[dict]) -> dict:
    c = defaultdict(int)
    for s in states:
        for a in s["attempts"]:
            c[a["failure"] or "ok"] += 1
    return dict(c)


def main():
    import argparse
    from .backends import make_backend
    from .data import get_eval_set
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="replay", choices=["replay", "vllm", "hf"])
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--per_bucket", type=int, default=10)
    ap.add_argument("--max_budget", type=int, default=50)
    ap.add_argument("--history_window", type=int, default=2)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--fail_first", type=int, default=0, help="replay backend only")
    ap.add_argument("--checker", default="gurobi", choices=["auto", "gurobi", "connectivity"])
    ap.add_argument("--out", default="runs/c0.jsonl")
    a = ap.parse_args()
    recs = get_eval_set(a.per_bucket)
    if a.backend == "replay":
        be = make_backend("replay", fail_first=a.fail_first)
    else:
        be = make_backend(a.backend, model=a.model, tensor_parallel_size=a.tp, temperature=a.temperature,
                          max_tokens=a.max_tokens, max_model_len=a.max_model_len)
    ug = {"auto": "auto", "gurobi": True, "connectivity": False}[a.checker]
    states = run_eval(recs, be, a.max_budget, use_gurobi=ug, history_window=a.history_window, out_path=a.out)
    print(json.dumps(success_curve(states), indent=1))
    print(json.dumps(failure_breakdown(states)))
    print(f"mean seconds/sample: {sum(s['seconds'] for s in states) / len(states):.2f}")


if __name__ == "__main__":
    main()
