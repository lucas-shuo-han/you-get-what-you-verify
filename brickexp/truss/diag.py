"""§4j diagnosis (exploratory): failure mechanism of misses. Seed-0 models are re-built deterministically
(same split / seed / threads) because run.py did not save weights; reproduction is checked against the logged miss.
python -m brickexp.truss.diag --fams howe pratt fink warren
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..screen.conformal import threshold
from .fem import E, SIGMA_ALLOW, solve
from .run import load, split, train, scores

OUT = Path("results/truss/diag"); OUT.mkdir(parents=True, exist_ok=True)


def mode(r):
    return max(r["util"], key=r["util"].get)


def governing(r):
    """member class of the governing member (by max of stress/buckling utilisation)."""
    X = np.array(r["X"]); M = np.array(r["M"]); A = np.array(r["A"]); I = np.array(r["I"])
    u, N = solve(X, M, A, np.array(r["fixed"], bool), np.array(r["F"]))
    d = X[M[:, 1]] - X[M[:, 0]]; L = np.linalg.norm(d, axis=1)
    st = np.abs(N) / A / SIGMA_ALLOW
    bk = np.where(N < 0, -N / (np.pi ** 2 * E * I / L ** 2), 0)
    k = int(np.argmax(np.maximum(st, bk)))
    vert = 2 if r["fam"] == "space" else 1
    h = abs(d[k, vert]) / L[k]
    geo = "chord" if h < 0.05 else ("vertical" if h > 0.95 else "diagonal")
    return f"{'comp' if N[k] < 0 else 'tens'}_{geo}", ("buckling" if bk[k] > st[k] else "stress")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fams", nargs="+"); ap.add_argument("--threads", type=int, default=7)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    R = load("data/truss"); res = {}
    small = [r for r in R if r["size"] == "small"]
    # 3. training-pool composition by controlling mode / governing member (all small, per family)
    comp = {}
    for fam in sorted({r["fam"] for r in small}):
        U = [r for r in small if r["fam"] == fam and r["unsafe"]]
        g = Counter(f"{governing(r)[0]}|{governing(r)[1]}" for r in U)
        comp[fam] = {"n": sum(r["fam"] == fam for r in small), "n_unsafe": len(U),
                     "mode": {k: v / len(U) for k, v in Counter(mode(r) for r in U).items()},
                     "governing": {k: v / len(U) for k, v in g.most_common()}}
    res["composition"] = comp
    for fam in a.fams:
        sh = f"loco_{fam}"
        tr, va, ca, pool, te = split(R, sh)
        log = lambda s: None
        model, _ = train(tr, va, 0, 20, "cpu", log)
        tau = threshold(scores(model, ca, "cpu"), np.array([r["unsafe"] for r in ca]), 0.05)
        s = scores(model, te, "cpu")
        U = [r for r, x in zip(te, s) if r["unsafe"]]
        miss = [r for r, x in zip(te, s) if r["unsafe"] and x < tau]
        logged = json.load(open(f"results/truss/{sh}.json"))["seeds"]["0"]["alpha_0.05"]["N0"]["miss"]
        gm = Counter(f"{governing(r)[0]}|{governing(r)[1]}" for r in miss)
        gu = Counter(f"{governing(r)[0]}|{governing(r)[1]}" for r in U)
        res[fam] = {"miss_rebuilt": len(miss) / len(U), "miss_logged": logged, "n_unsafe": len(U), "n_miss": len(miss),
                    "missed_mode": {k: v / max(1, len(miss)) for k, v in Counter(mode(r) for r in miss).items()},
                    "unsafe_mode": {k: v / len(U) for k, v in Counter(mode(r) for r in U).items()},
                    "missed_any_mode": {m: float(np.mean([m in r["modes"] for r in miss])) if miss else None
                                        for m in ("stress", "buckling", "disp")},
                    "missed_governing": {k: v / max(1, len(miss)) for k, v in gm.most_common()},
                    "unsafe_governing": {k: v / len(U) for k, v in gu.most_common()}}
        print(fam, json.dumps({k: res[fam][k] for k in ("miss_rebuilt", "miss_logged", "n_miss")}), flush=True)
    (OUT / "diag.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
