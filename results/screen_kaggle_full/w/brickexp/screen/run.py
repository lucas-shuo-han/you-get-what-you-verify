"""§4g end-to-end: train screener (per seed), conformal-calibrate, evaluate ID / held-out category / larger size,
recalibration curve, and apply the locked §4g decision rules.

Splits (by structure_id hash, so no structure is shared across parts):
  id.jsonl   -> train 70 / val 10 / cal 10 / test 10
  cat.jsonl  -> per held-out category: recal pool 40% / test 60%   (N labels drawn from the pool)
  size.jsonl -> test only (calibrated with the ID calibration set, N=0)

Usage: python -m brickexp.screen.run --data data/screen --out results/screen --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .conformal import threshold, metrics, id_tolerance
from .labels import HELD
from .gnn import Screener, brick_graph, collate, parse_text

ALPHAS = [0.01, 0.05]
NS = [0, 10, 30, 100, 300, 1000]


def _u(sid, salt="split"):
    return int(hashlib.md5(f"{salt}-{sid}".encode()).hexdigest()[:8], 16) / 2 ** 32


def load(path):
    rows, n_err = [], 0
    for l in open(path):
        r = json.loads(l)
        if r["unstable"] is None:
            n_err += 1
            continue
        r["g"] = brick_graph(parse_text(r["text"]))
        rows.append(r)
    rows.sort(key=lambda r: r["i"])
    return rows, n_err


def _dev(b, dev):
    return tuple(x.to(dev) if torch.is_tensor(x) else x for x in b)


def scores_of(model, rows, bs=256):
    model.eval(); out = []; dev = next(model.parameters()).device
    with torch.no_grad():
        for k in range(0, len(rows), bs):
            out.append(torch.sigmoid(model(*_dev(collate([r["g"] for r in rows[k:k + bs]]), dev))).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train(tr, va, seed, epochs, bs=64, lr=2e-3, H=64, L=4, max_steps=0, log=print, dev="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Screener(H, L).to(dev)
    log(f"model H={H} L={L} params={sum(p.numel() for p in model.parameters())} device={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    y = torch.tensor([r["unstable"] for r in tr], dtype=torch.float32, device=dev)
    yv = np.array([r["unstable"] for r in va])
    best, best_state, step = 1e9, None, 0
    for ep in range(epochs):
        model.train(); t = time.time()
        for idx in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // bs)):
            loss = F.binary_cross_entropy_with_logits(model(*_dev(collate([tr[i]["g"] for i in idx]), dev)), y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if max_steps and step >= max_steps:
                break
        sv = np.clip(scores_of(model, va), 1e-6, 1 - 1e-6)
        vl = float(-np.mean(yv * np.log(sv) + (1 - yv) * np.log(1 - sv)))
        log(f"seed {seed} ep {ep} val_bce {vl:.4f} {time.time() - t:.0f}s")
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if max_steps and step >= max_steps:
            break
    model.load_state_dict(best_state)
    return model, best


def verdict(id_m, shift_m, recal, alpha):
    """§4g locked rules for one shift and alpha. id_m/shift_m: N=0 metric dicts; recal: {N: mean metric dict}."""
    drop = 1 - shift_m["saved"] / id_m["saved"] if id_m["saved"] > 0 else float("nan")
    real = shift_m["miss"] >= 2 * alpha or drop >= 0.5
    r = recal.get(1000) if recal else None
    recovered = None if r is None else (r["miss"] <= alpha and r["saved"] > 0)
    if real:
        v = "REAL_fixable" if recovered else ("REAL_unfixable" if recovered is False else "REAL(recal n/a)")
    elif shift_m["miss"] <= alpha and abs(drop) < 0.5:
        v = "NOT_REAL"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "shift_miss": shift_m["miss"], "id_saved": id_m["saved"], "shift_saved": shift_m["saved"],
            "saved_drop": drop, "N1000": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/screen")
    ap.add_argument("--out", default="results/screen")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    d = Path(a.data)
    idr, e_id = load(d / "id.jsonl")
    parts = {"train": [], "val": [], "cal": [], "test": []}
    for r in idr:
        u = _u(r["sid"])
        parts["train" if u < .7 else "val" if u < .8 else "cal" if u < .9 else "test"].append(r)
    for k in parts:  # §4g: held-out categories never in train / val / ID calibration / ID test
        assert not any(r["cat"] in HELD for r in parts[k]), f"held-out category leaked into {k}"
    shifts, n_err = {}, {"id": e_id}
    if (d / "cat.jsonl").exists():
        cr, n_err["cat"] = load(d / "cat.jsonl")
        for c in sorted({r["cat"] for r in cr}):
            rc = [r for r in cr if r["cat"] == c]
            shifts[f"cat_{c}"] = {"pool": [r for r in rc if _u(r["sid"], "recal") < .4],
                                  "test": [r for r in rc if _u(r["sid"], "recal") >= .4]}
    if (d / "size.jsonl").exists():
        sr, n_err["size"] = load(d / "size.jsonl")
        shifts["size"] = {"pool": [], "test": sr}
    log(json.dumps({k: len(v) for k, v in parts.items()} | {s: {k: len(v) for k, v in p.items()} for s, p in shifts.items()}
                   | {"solver_errors": n_err}))

    res = {"seeds": {}}
    for seed in a.seeds:
        model, vbce = train(parts["train"], parts["val"], seed, a.epochs, bs=a.bs, lr=a.lr, H=a.H, L=a.L,
                             max_steps=a.max_steps, log=log, dev=dev)
        lab = lambda rows: np.array([r["unstable"] for r in rows])
        sc = {k: scores_of(model, parts[k]) for k in ("cal", "test")}
        R = {"val_bce": vbce}
        for alpha in ALPHAS:
            tau = threshold(sc["cal"], lab(parts["cal"]), alpha)
            A = {"id": metrics(sc["test"], lab(parts["test"]), tau)}
            A["id"]["tol"] = id_tolerance(alpha, int(lab(parts["cal"]).sum()), A["id"]["n_unstable"])
            if not a.max_steps:  # §4g sanity: stop and investigate if ID miss breaks the guarantee
                assert A["id"]["miss"] <= alpha + A["id"]["tol"], (alpha, A["id"])
            for s, p in shifts.items():
                st, ytest = scores_of(model, p["test"]), lab(p["test"])
                A[s] = {"N0": metrics(st, ytest, tau)}
                if p["pool"]:
                    sp, yp = scores_of(model, p["pool"]), lab(p["pool"])
                    rng = np.random.default_rng(1000 + seed)
                    for N in NS[1:]:
                        if N > len(sp):
                            continue
                        new, pooled = [], []
                        for _ in range(a.draws):
                            j = rng.choice(len(sp), N, replace=False)
                            new.append(metrics(st, ytest, threshold(sp[j], yp[j], alpha)))
                            pooled.append(metrics(st, ytest, threshold(np.r_[sc["cal"], sp[j]],
                                                                       np.r_[lab(parts["cal"]), yp[j]], alpha)))
                        agg = lambda L: {k: float(np.mean([m[k] for m in L])) for k in ("miss", "saved", "unsafe_among_safe")} \
                            | {"miss_p95": float(np.percentile([m["miss"] for m in L], 95)),
                               "frac_draws_miss_gt_alpha": float(np.mean([m["miss"] > alpha for m in L])),
                               "frac_draws_infeasible": float(np.mean([m["infeasible"] for m in L]))}
                        A[s][f"N{N}"] = agg(new)
                        A[s][f"N{N}_pooled"] = agg(pooled)
            R[f"alpha_{alpha}"] = A
            log(f"seed {seed} alpha {alpha} id miss {A['id']['miss']:.4f} saved {A['id']['saved']:.3f} auc {A['id']['auc']:.3f} | "
                + " | ".join(f"{s} miss {A[s]['N0']['miss']:.4f} saved {A[s]['N0']['saved']:.3f} auc {A[s]['N0']['auc']:.3f}" for s in shifts))
        res["seeds"][seed] = R
        torch.save(model.state_dict(), out / f"screener_s{seed}.pt")

    # aggregate over seeds + §4g verdicts
    summ = {}
    for alpha in ALPHAS:
        key = f"alpha_{alpha}"
        S = [res["seeds"][s][key] for s in a.seeds]
        mean = lambda f: float(np.mean([f(x) for x in S]))
        rg = lambda f: [float(np.min([f(x) for x in S])), float(np.max([f(x) for x in S]))]
        KS = ("miss", "saved", "auc", "precision_safe")
        idm = {k: mean(lambda x: x["id"][k]) for k in KS} | {k + "_range": rg(lambda x: x["id"][k]) for k in KS}
        summ[key] = {"id": idm}
        for s in shifts:
            sh = {k: mean(lambda x: x[s]["N0"][k]) for k in KS} | {k + "_range": rg(lambda x: x[s]["N0"][k]) for k in KS}
            recal = {N: {k: mean(lambda x: x[s][f"N{N}"][k]) for k in ("miss", "saved", "miss_p95", "frac_draws_infeasible")}
                     for N in NS[1:] if f"N{N}" in S[0][s]}
            summ[key][s] = {"N0": sh, "recal": recal, **verdict(idm, sh, recal, alpha)}
    summ["n_seeds"] = len(a.seeds); summ["model"] = {"H": a.H, "L": a.L}
    res["summary"] = summ
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    log(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
