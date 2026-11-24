"""Analysis + main figure: success vs checker budget, one line per condition, 95% bootstrap CI.

Usage:
  python -m brickexp.analysis --run C0=results/c0/*.jsonl --run C1=results/c1/s*/*.jsonl --out results/fig_budget.png
Multiple files per condition (seeds) are pooled per prompt: CI resamples prompts, averaging over seeds.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

from .data import load_jsonl

BUDGETS = (0, 1, 10, 50)
BUCKETS = ("small", "medium", "large")
LOCAL = {"format", "out_of_bounds", "collision", "floating"}
GLOBAL = {"unstable"}


def solved_matrix(runs: list[list[dict]], budgets=BUDGETS) -> tuple[np.ndarray, list[str], list[int]]:
    """-> (n_prompts x len(budgets)) success averaged over seeds, bucket per prompt, usable budgets."""
    mb = min(s["max_budget"] for r in runs for s in r)
    ks = [k for k in budgets if k <= mb]
    by_id = defaultdict(list)
    bucket = {}
    for r in runs:
        for s in r:
            by_id[s["structure_id"]].append([s["solved_at"] is not None and s["solved_at"] <= k for k in ks])
            bucket[s["structure_id"]] = s["bucket"]
    ids = sorted(by_id)
    return np.array([np.mean(by_id[i], axis=0) for i in ids]), [bucket[i] for i in ids], ks


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """x: (n, K) -> mean (K,), lo (K,), hi (K,) via prompt-level bootstrap."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    return x.mean(0), np.quantile(boots, alpha / 2, axis=0), np.quantile(boots, 1 - alpha / 2, axis=0)


def summarize(runs, budgets=BUDGETS) -> dict:
    M, buckets, ks = solved_matrix(runs, budgets)
    out = {}
    for g in ("all",) + BUCKETS:
        sel = np.array([g == "all" or b == g for b in buckets])
        if sel.sum() == 0:
            continue
        m, lo, hi = bootstrap_ci(M[sel])
        out[g] = {k: {"mean": float(m[j]), "lo": float(lo[j]), "hi": float(hi[j]), "n": int(sel.sum())}
                  for j, k in enumerate(ks)}
    return out


def error_locality(runs) -> dict:
    """Fraction of failed attempts that are local (collision/floating/...) vs global (unstable), by attempt index."""
    c = defaultdict(lambda: defaultdict(int))
    for r in runs:
        for s in r:
            for a in s["attempts"]:
                f = a["failure"]
                if f is None:
                    continue
                c[a["attempt"]]["local" if f in LOCAL else "global" if f in GLOBAL else "other"] += 1
    return {k: dict(v) for k, v in sorted(c.items())}


def timing(runs) -> dict:
    secs = [s["seconds"] for r in runs for s in r]
    att = [len(s["attempts"]) for r in runs for s in r]
    return {"mean_s_per_prompt": float(np.mean(secs)), "mean_attempts": float(np.mean(att)),
            "mean_s_per_attempt": float(np.sum(secs) / max(1, np.sum(att)))}


def plot(conds: dict[str, dict], out: str, groups=("all",) + BUCKETS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.4), sharey=True)
    for ax, g in zip(np.atleast_1d(axes), groups):
        for name, summ in conds.items():
            if g not in summ:
                continue
            ks = sorted(summ[g], key=int)
            xs = np.arange(len(ks))
            m = [summ[g][k]["mean"] for k in ks]
            lo = [summ[g][k]["lo"] for k in ks]
            hi = [summ[g][k]["hi"] for k in ks]
            ax.plot(xs, m, marker="o", label=name)
            ax.fill_between(xs, lo, hi, alpha=0.2)
            ax.set_xticks(xs, [str(k) for k in ks])
        n = next((v[next(iter(v))]["n"] for v in (s.get(g) for s in conds.values()) if v), 0)
        ax.set_title(f"{g} (n={n})")
        ax.set_xlabel("checker budget k (retries)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("success rate")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=glob (repeatable)")
    ap.add_argument("--out", default="results/fig_budget.png")
    a = ap.parse_args()
    conds, report = {}, {}
    for spec in a.run:
        name, pat = spec.split("=", 1)
        runs = [load_jsonl(f) for f in sorted(glob.glob(pat))]
        if not runs:
            raise SystemExit(f"no files for {spec}")
        conds[name] = summarize(runs)
        report[name] = {"curve": conds[name], "locality": error_locality(runs), "timing": timing(runs),
                        "n_seeds": len(runs)}
    print(json.dumps(report, indent=1))
    with open(a.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(report, f, indent=1)
    print("figure:", plot(conds, a.out))


if __name__ == "__main__":
    main()
