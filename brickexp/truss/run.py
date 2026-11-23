"""§4j screener on trusses: same small message-passing GNN (<=100k params), one-sided conformal, LOCO + size shift.

Graph: node = pin joint; edge = member (both directions).
Node feats (9): x/S, y/S, z/S, fixed xyz, load xyz / 1e5 N.  Edge feats (6): L/10 m, A*1e3, L/r/200, |direction cosines|.
Global: span/40 appended to the readout.

Shifts (§4j, alpha in {1%,5%}, N grid, 3 seeds):
  loco_<fam> : train/val/cal on the other 7 families (small panels), test fam: recal pool 40% / test 60%
  size       : train/val/cal on all 8 families (small panels), test = large panels (pool 40% / test 60%)
Early warning (locked definition, see DIRECTION §4j lock): outlier rate = frac of target samples with
  span_depth > p95(train) OR max_slender > p95(train), percentiles from that run's own training set.
Usage: python -m brickexp.truss.run --shift loco_pratt --seeds 0 1 2 --out results/truss
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..screen.conformal import threshold, metrics, id_tolerance
from .gen import FAMILIES10 as FAMILIES

ALPHAS = [0.01, 0.05]
NS = [0, 10, 30, 100, 300, 1000]
NF, EF = 9, 6


def _u(r, salt):
    return int(hashlib.md5(f"{salt}-{r['fam']}-{r['size']}-{r['i']}".encode()).hexdigest()[:8], 16) / 2 ** 32


def graph(r):
    X = np.array(r["X"]); M = np.array(r["M"]); A = np.array(r["A"]); I = np.array(r["I"]); S = r["span"]
    d = X[M[:, 1]] - X[M[:, 0]]; L = np.linalg.norm(d, axis=1)
    nf = np.c_[X / S, np.array(r["fixed"]), np.array(r["F"]) / 1e5].astype(np.float32)
    ef = np.c_[L / 10, A * 1e3, L / np.sqrt(I / A) / 200, np.abs(d) / L[:, None]].astype(np.float32)
    e = np.r_[M, M[:, ::-1]].astype(np.int64)
    return nf, e, np.r_[ef, ef], np.float32(S / 40)


def collate(gs, dev):
    nf, es, ef, gi, sp, off = [], [], [], [], [], 0
    for g, (a, e, w, s) in enumerate(gs):
        nf.append(a); es.append(e + off); ef.append(w); gi.append(np.full(len(a), g)); sp.append(s); off += len(a)
    t = lambda v, dt: torch.from_numpy(np.concatenate(v) if isinstance(v, list) else v).to(dev, dt)
    return t(nf, torch.float32), t(es, torch.int64), t(ef, torch.float32), t(gi, torch.int64), \
        torch.tensor(sp, device=dev)[:, None], len(gs)


class Layer(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * H + EF, H), nn.GELU(), nn.Linear(H, H))
        self.upd = nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Linear(H, H))
        self.ln = nn.LayerNorm(H)

    def forward(self, h, e, ef):
        s, d = e[:, 0], e[:, 1]
        m = self.msg(torch.cat([h[s], h[d], ef], 1))
        agg = torch.zeros_like(h).index_add_(0, d, m)
        deg = torch.zeros(len(h), device=h.device).index_add_(0, d, torch.ones_like(d, dtype=h.dtype)).clamp(min=1)
        mx = torch.full_like(h, -1e9).scatter_reduce(0, d[:, None].expand(-1, h.shape[1]), m, "amax")
        mx = torch.where(mx < -1e8, torch.zeros_like(mx), mx)
        return self.ln(h + self.upd(torch.cat([h, agg / deg[:, None], mx], 1)))


class Screener(nn.Module):
    """-> logit P(unsafe). Max-pooling is kept so a single critical member can dominate."""
    def __init__(self, H=48, L=4, aux=False):
        super().__init__()
        self.aux = nn.Sequential(nn.Linear(H + EF, H), nn.GELU(), nn.Linear(H, 1)) if aux else None
        self.emb = nn.Sequential(nn.Linear(NF, H), nn.GELU(), nn.Linear(H, H))
        self.layers = nn.ModuleList([Layer(H) for _ in range(L)])
        self.head = nn.Sequential(nn.Linear(2 * H + 1, H), nn.GELU(), nn.Linear(H, 1))

    def forward(self, nf, e, ef, gi, sp, G):
        h = self.emb(nf)
        for l in self.layers:
            h = l(h, e, ef)
        self._last = (h, e, ef)
        H = h.shape[1]
        cnt = torch.zeros(G, device=h.device).index_add_(0, gi, torch.ones_like(gi, dtype=h.dtype))
        mean = torch.zeros(G, H, device=h.device).index_add_(0, gi, h) / cnt[:, None]
        mx = torch.full((G, H), -1e9, device=h.device).scatter_reduce(0, gi[:, None].expand(-1, H), h, "amax")
        return self.head(torch.cat([mean, mx, sp], 1)).squeeze(1)


    def edge_logits(self):
        """§4k aux: per-edge logit P(member in compression); symmetric in the two directions."""
        h, e, ef = self._last
        return self.aux(torch.cat([h[e[:, 0]] + h[e[:, 1]], ef], 1)).squeeze(1)


def edge_comp(model, rs, dev, bs=256):
    """-> list (per sample) of predicted P(compression) per member (forward-direction edges)."""
    model.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(rs), bs):
            chunk = rs[k:k + bs]; model(*collate([r["g"] for r in chunk], dev))
            p = torch.sigmoid(model.edge_logits()).cpu().numpy(); o = 0
            for r in chunk:
                m = len(r["M"]); out.append(p[o:o + m]); o += 2 * m
    return out


def scores(model, rs, dev, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(rs), bs):
            out.append(torch.sigmoid(model(*collate([r["g"] for r in rs[k:k + bs]], dev))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train(tr, va, seed, epochs, dev, log, bs=64, lr=2e-3, max_steps=0, aux=False, aux_w=0.5):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Screener(aux=aux).to(dev)
    log(f"params={sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    y = torch.tensor([r["unsafe"] for r in tr], dtype=torch.float32, device=dev); yv = np.array([r["unsafe"] for r in va])
    best, state, step = 1e9, None, 0
    for ep in range(epochs):
        model.train(); t = time.time()
        for idx in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // bs)):
            loss = F.binary_cross_entropy_with_logits(model(*collate([tr[i]["g"] for i in idx], dev)), y[idx])
            if aux:  # supervision: FEM member force sign of the training families (both edge directions)
                c = torch.from_numpy(np.concatenate([np.r_[tr[i]["comp"], tr[i]["comp"]] for i in idx]).astype(np.float32)).to(dev)
                loss = loss + aux_w * F.binary_cross_entropy_with_logits(model.edge_logits(), c)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if max_steps and step >= max_steps:
                break
        sched.step()
        sv = np.clip(scores(model, va, dev), 1e-6, 1 - 1e-6)
        vl = float(-np.mean(yv * np.log(sv) + (1 - yv) * np.log(1 - sv)))
        log(f"seed {seed} ep {ep} val_bce {vl:.4f} {time.time() - t:.0f}s")
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if max_steps and step >= max_steps:
            break
    model.load_state_dict(state)
    return model, best


def load(d):
    R = []
    for f in ("small", "large"):
        for l in open(Path(d) / f"{f}.jsonl"):
            r = json.loads(l); r["g"] = graph(r)
            X = np.array(r["X"]); M = np.array(r["M"])
            r["max_sl_all"] = np.linalg.norm(X[M[:, 1]] - X[M[:, 0]], axis=1) / np.sqrt(np.array(r["I"]) / np.array(r["A"]))
            R.append(r)
    return R


def split(R, shift):
    small = [r for r in R if r["size"] == "small"]
    if shift == "size":
        src, tgt = small, [r for r in R if r["size"] == "large"]
    else:
        fam = shift.removeprefix("loco_"); assert fam in FAMILIES
        src, tgt = [r for r in small if r["fam"] != fam], [r for r in small if r["fam"] == fam]
    tr = [r for r in src if _u(r, "split") < .7]; va = [r for r in src if .7 <= _u(r, "split") < .8]
    ca = [r for r in src if _u(r, "split") >= .8]
    pool = [r for r in tgt if _u(r, "recal") < .4]; te = [r for r in tgt if _u(r, "recal") >= .4]
    if shift != "size":
        assert not any(r["fam"] == shift[5:] for r in tr + va + ca), "held-out family leaked"
    return tr, va, ca, pool, te


def outlier_rate(tr, te):
    p_sd = np.percentile([r["span_depth"] for r in tr], 95); p_sl = np.percentile([r["max_slender"] for r in tr], 95)
    f = lambda rs: float(np.mean([(r["span_depth"] > p_sd) or (r["max_slender"] > p_sl) for r in rs]))
    return {"outlier_rate": f(te), "outlier_rate_train": f(tr), "p95_span_depth": float(p_sd), "p95_slender": float(p_sl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data", default="data/truss"); ap.add_argument("--out", default="results/truss")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--threads", type=int, default=6); ap.add_argument("--aux", action="store_true")
    ap.add_argument("--extra", help="§4k round 5: extra TRAIN-only jsonl in <data>/ (e.g. aug_flip)"); ap.add_argument("--max_steps", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = a.shift + ("_aux" if a.aux else "") + (f"_{a.extra}" if a.extra else "")
    logf = open(out / f"{tag}.log", "a")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    tr, va, ca, pool, te = split(load(a.data), a.shift)
    if a.extra:
        ex = []
        for l in open(Path(a.data) / f"{a.extra}.jsonl"):
            r = json.loads(l); r["g"] = graph(r); ex.append(r)
        assert not any(r["fam"] == a.shift.removeprefix("loco_") for r in ex)
        tr = tr + ex
    lab = lambda rs: np.array([r["unsafe"] for r in rs])
    log(json.dumps({"shift": a.shift, "train": len(tr), "val": len(va), "cal": len(ca), "pool": len(pool), "test": len(te),
                    "unsafe_train": float(lab(tr).mean()), "unsafe_test": float(lab(te).mean())}))
    res = {"shift": a.shift, "signal": outlier_rate(tr, pool + te), "seeds": {}}
    for seed in a.seeds:
        model, vb = train(tr, va, seed, a.epochs, dev, log, max_steps=a.max_steps, aux=a.aux)
        torch.save(model.state_dict(), out / f"{tag}_s{seed}.pt")
        sc_ca, sc_te, sc_po = scores(model, ca, dev), scores(model, te, dev), scores(model, pool, dev)
        sc_id = scores(model, [r for r in va], dev)  # ID check on val (never used for tau)
        R = {"val_bce": vb}
        from ..screen.conformal import auc as _auc
        R["id_val_auc"] = _auc(sc_id, lab(va))
        np.savez(out / f"{tag}_s{seed}_scores.npz", ca=sc_ca, ya=lab(ca), po=sc_po, yp=lab(pool), te=sc_te, yt=lab(te),
                 va=sc_id, yv=lab(va), fv=np.array([r["fam"] for r in va]))
        if a.aux:  # §4k round 4 signals (predictions only; no target labels, no solver)
            def web_comp_frac(rs):
                P = edge_comp(model, rs, dev)
                return np.array([float(np.mean([p > .5 for p, g in zip(pp, r["grp"]) if g == "web"] or [0])) for pp, r in zip(P, rs)]), P
            ftr, Ptr = web_comp_frac(tr); fte, Pte = web_comp_frac(pool + te)
            sl = lambda rs, P: np.concatenate([np.asarray(r["max_sl_all"])[p > .5] for r, p in zip(rs, P)])
            from scipy.stats import ks_2samp
            R["signal_mech"] = {"primary": float(abs(fte.mean() - ftr.mean())), "web_comp_frac_target": float(fte.mean()),
                                "web_comp_frac_train": float(ftr.mean()),
                                "ks_slender": float(ks_2samp(sl(tr, Ptr), sl(pool + te, Pte)).statistic),
                                "aux_acc_target": float(np.mean(np.concatenate([(p > .5) == np.asarray(r["comp"], bool) for r, p in zip(pool + te, Pte)])))}
            log(json.dumps(R["signal_mech"]))
        for alpha in ALPHAS:
            tau = threshold(sc_ca, lab(ca), alpha)
            A = {"id_val": metrics(sc_id, lab(va), tau), "N0": metrics(sc_te, lab(te), tau)}
            y0 = lab(te) == 0  # §4j lock: pass rate = judged safe / truly safe (decision metric; 'saved' report only)
            A["N0"]["pass_rate"] = float((sc_te[y0] < tau).mean()) if y0.any() else float("nan")
            A["id_val"]["tol"] = id_tolerance(alpha, int(lab(ca).sum()), A["id_val"]["n_unstable"])
            if not a.max_steps:
                assert A["id_val"]["miss"] <= alpha + A["id_val"]["tol"], ("ID guarantee broken", alpha, A["id_val"])
            rng = np.random.default_rng(1000 + seed); yp = lab(pool)
            for N in NS[1:]:
                if N > len(pool):
                    continue
                new, pooled = [], []
                for _ in range(a.draws):
                    j = rng.choice(len(pool), N, replace=False)
                    new.append(metrics(sc_te, lab(te), threshold(sc_po[j], yp[j], alpha)))
                    pooled.append(metrics(sc_te, lab(te), threshold(np.r_[sc_ca, sc_po[j]], np.r_[lab(ca), yp[j]], alpha)))
                agg = lambda L: {k: float(np.mean([m[k] for m in L])) for k in ("miss", "saved")} | {
                    "miss_p95": float(np.percentile([m["miss"] for m in L], 95)),
                    "frac_draws_infeasible": float(np.mean([m["infeasible"] for m in L]))}
                A[f"N{N}"] = agg(new); A[f"N{N}_pooled"] = agg(pooled)
            R[f"alpha_{alpha}"] = A
            log(f"{a.shift} seed {seed} a={alpha} id_val miss {A['id_val']['miss']:.4f} | N0 miss {A['N0']['miss']:.4f} "
                f"saved {A['N0']['saved']:.3f} pass {A['N0']['pass_rate']:.3f} auc {A['N0']['auc']:.3f}")
        res["seeds"][seed] = R
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
