"""§4i part 2 (locked, DIRECTION.md §4i): leave-one-category-out, small model, 3 seeds.

Per target category c (all categories except guitar = dev):
  train/val/cal = all labelled samples of the other categories, guitar excluded (split by structure hash 70/10/20;
                  train capped at 14k by hash order to match the §4g small-model regime)
  test          = every labelled sample of c (sets id + cat + loco)
  record        = miss and miss excess at alpha=0.05, N=0 (ID calibration), saved, AUC
  signals (label-free on c; percentiles from THIS run's training set):
      S2 (primary) = frac of c with com_offset > train p95
      S1           = frac of c with aspect > train p95
      S3           = MMD^2 (RBF, median heuristic) of mean-pooled last-layer embeddings, 1000 vs 1000, fixed seed
Usage: python -m brickexp.screen.loco run --cats <c ...> --out results/screen_loco   (resumable per cat/seed)
       python -m brickexp.screen.loco summary --out results/screen_loco
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .conformal import threshold, metrics
from .diag import phys, rows, GUITAR
from .gnn import brick_graph, collate, parse_text
from .run import _u, _dev, scores_of, train

ALPHA = 0.05
TRAIN_CAP = 14000
EPOCHS, MAX_STEPS = 15, 0


def embed(model, rs, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(rs), bs):
            feat, e, ew, gi, cand, G = collate([r["g"] for r in rs[k:k + bs]])
            h = model.emb(feat)
            for l in model.layers:
                h = l(h, e, ew)
            cnt = torch.zeros(G).index_add_(0, gi, torch.ones_like(gi, dtype=h.dtype))
            out.append((torch.zeros(G, h.shape[1]).index_add_(0, gi, h) / cnt[:, None]).numpy())
    return np.concatenate(out)


def mmd2(X, Y):
    Z = np.r_[X, Y]
    d = ((Z[:, None] - Z[None]) ** 2).sum(-1)
    bw = np.median(((X[:, None] - X[None]) ** 2).sum(-1)[np.triu_indices(len(X), 1)])
    K = np.exp(-d / max(bw, 1e-12)); n = len(X)
    kxx, kyy, kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    m = len(Y)
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())


def load_all():
    R = [r for r in rows() if r["cat"] != GUITAR]
    for r in R:
        b = parse_text(r["text"]); r.update(phys(b)); r["g"] = brick_graph(b)
    return R


def run_one(R, c, seed, out, log):
    f = out / f"{c}_s{seed}.json"
    if f.exists():
        return
    other = [r for r in R if r["cat"] != c]
    test = [r for r in R if r["cat"] == c]
    u = {id(r): _u(r["sid"], "loco") for r in other}
    tr = sorted([r for r in other if u[id(r)] < .7], key=lambda r: u[id(r)])[:TRAIN_CAP]
    va = [r for r in other if .7 <= u[id(r)] < .8]
    ca = [r for r in other if u[id(r)] >= .8]
    assert not any(r["cat"] in (c, GUITAR) for r in tr + va + ca)
    t = time.time()
    model, vb = train(tr, va, seed, EPOCHS, max_steps=MAX_STEPS, log=log)
    lab = lambda rs: np.array([r["unstable"] for r in rs])
    tau = threshold(scores_of(model, ca), lab(ca), ALPHA)
    m = metrics(scores_of(model, test), lab(test), tau)
    sig = {}
    for k, name in (("com_offset", "S2"), ("aspect", "S1")):
        p95 = np.percentile([r[k] for r in tr], 95)
        sig[name] = float(np.mean([r[k] > p95 for r in test]))
    rng = np.random.default_rng(seed)
    pick = lambda rs: [rs[i] for i in rng.choice(len(rs), min(1000, len(rs)), replace=False)]
    sig["S3"] = mmd2(embed(model, pick(tr)), embed(model, pick(test)))
    res = {"cat": c, "seed": seed, "n_train": len(tr), "n_test": len(test), "val_bce": vb, "tau": tau,
           **{k: m[k] for k in ("miss", "saved", "auc", "precision_safe", "n_unstable", "infeasible")},
           "miss_excess": m["miss"] - ALPHA, **sig, "sec": round(time.time() - t)}
    f.write_text(json.dumps(res, default=float))
    log(json.dumps(res, default=float))


def _auroc(x, y):
    from .conformal import auc
    return auc(np.asarray(x, float), np.asarray(y, bool))


def _spearman(x, y):
    from .conformal import _rank_avg
    a, b = _rank_avg(np.asarray(x, float)), _rank_avg(np.asarray(y, float))
    return float(np.corrcoef(a, b)[0, 1])


def summary(out):
    rs = [json.loads(p.read_text()) for p in sorted(out.glob("*_s*.json"))]
    cats = sorted({r["cat"] for r in rs})
    T = []
    for c in cats:
        L = [r for r in rs if r["cat"] == c]
        row = {"cat": c, "seeds": len(L), "n_test": L[0]["n_test"], "n_unstable": L[0]["n_unstable"]}
        for k in ("miss", "miss_excess", "saved", "auc", "S1", "S2", "S3"):
            v = [r[k] for r in L]; row[k] = float(np.mean(v)); row[k + "_range"] = [float(min(v)), float(max(v))]
        row["positive"] = row["miss"] >= 2 * ALPHA
        T.append(row)
    pos = [r["positive"] for r in T]
    V = {"n_cats": len(T), "n_positive": int(sum(pos))}
    if sum(pos) >= 3 and sum(pos) < len(T):
        V["AUROC"] = {s: _auroc([r[s] for r in T], pos) for s in ("S2", "S1", "S3")}
        V["verdict"] = "EFFECTIVE" if V["AUROC"]["S2"] >= 0.8 else "NOT_EFFECTIVE"
    else:
        V["verdict"] = "FAILURE_RARE" if sum(pos) < 3 else "ALL_POSITIVE"
    V["spearman_vs_miss_excess"] = {s: _spearman([r[s] for r in T], [r["miss_excess"] for r in T]) for s in ("S2", "S1", "S3")}
    (out / "summary.json").write_text(json.dumps({"verdict": V, "table": T}, indent=1))
    print(json.dumps(V, indent=1))
    print("| cat | n | n_unst | miss (range) | saved | AUC | S2 | S1 | S3 | pos |\n|" + "---|" * 10)
    for r in sorted(T, key=lambda r: -r["miss"]):
        print(f"| {r['cat']} | {r['n_test']} | {r['n_unstable']} | {r['miss']:.3f} ({r['miss_range'][0]:.3f}-{r['miss_range'][1]:.3f}) | "
              f"{r['saved']:.3f} | {r['auc']:.3f} | {r['S2']:.3f} | {r['S1']:.3f} | {r['S3']:.4f} | {'Y' if r['positive'] else ''} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--cats", nargs="*"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/screen_loco"); ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global EPOCHS, MAX_STEPS
    if a.smoke:
        EPOCHS, MAX_STEPS = 1, 20
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "summary":
        return summary(out)
    torch.set_num_threads(a.threads)
    logf = open(out / f"run_{a.cats[0] if a.cats else 'all'}.log", "a")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    R = load_all()
    cats = a.cats or sorted({r["cat"] for r in R})
    for c in cats:
        for s in a.seeds:
            run_one(R, c, s, out, log)


if __name__ == "__main__":
    main()
