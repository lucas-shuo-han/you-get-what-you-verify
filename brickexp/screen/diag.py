"""§4i part 1 (exploratory): guitar vs other categories. Small model only; guitar is a dev set.

  python -m brickexp.screen.diag rerun --workers 4     # Gurobi per-brick scores of unstable samples (failure types)
  python -m brickexp.screen.diag report                # stats -> results/screen_diag/{diag.json,diag.md}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path("data/screen")
OUT = Path("results/screen_diag")
GUITAR = "03467517"


def rows():
    out = []
    for f in ("id", "cat", "loco"):
        p = D / f"{f}.jsonl"
        if p.exists():
            for l in p.open():
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                if r["unstable"] is not None:
                    r["set"] = f; out.append(r)
    return out


# ---------- label-free physics stats (bricks only) ----------
def phys(bricks):
    from .gnn import SBrick  # noqa
    a = np.array([[b.h, b.w, b.x, b.y, b.z] for b in bricks], float)
    h, w, x, y, z = a.T
    area = h * w
    occ = {}  # (x,y,z) -> brick idx
    for k, b in enumerate(bricks):
        for i in range(b.x, b.x + b.h):
            for j in range(b.y, b.y + b.w):
                occ[(i, j, b.z)] = k

    def overhang(k):
        b = bricks[k]
        if b.z == 0:
            return 0.0
        sup = sum((i, j, b.z - 1) in occ for i in range(b.x, b.x + b.h) for j in range(b.y, b.y + b.w))
        return 1 - sup / (b.h * b.w)

    oh = np.array([overhang(k) for k in range(len(bricks))])
    cx, cy = x + h / 2, y + w / 2
    m = area.sum()
    com = np.array([(cx * area).sum() / m, (cy * area).sum() / m])
    g = z == 0
    bx0, bx1, by0, by1 = x[g].min(), (x + h)[g].max(), y[g].min(), (y + w)[g].max()
    base_c = np.array([(bx0 + bx1) / 2, (by0 + by1) / 2]); base_r = max(bx1 - bx0, by1 - by0) / 2
    # candidate: moment arm of the candidate about the nearest support below (proxy for added torque)
    c = bricks[-1]
    return {"n": len(bricks), "height": int(z.max()) + 1,
            "aspect": (z.max() + 1) / max(1.0, np.sqrt(area[g].sum())),
            "base_area": float(area[g].sum()),
            "cand_overhang": float(oh[-1]), "cand_z": int(c.z), "cand_area": int(c.h * c.w),
            "mean_overhang": float(oh[z > 0].mean()) if (z > 0).any() else 0.0,
            "frac_overhang_gt_half": float((oh[z > 0] > .5).mean()) if (z > 0).any() else 0.0,
            "com_offset": float(np.linalg.norm(com - base_c) / max(base_r, 1)),
            "moment": float((area * np.linalg.norm(np.stack([cx, cy], 1) - base_c, axis=1) * (z + 1)).sum() / m)}


# ---------- failure types via Gurobi per-voxel scores ----------
def _rerun(r):
    from ..checker import gurobi_scores, parse
    from brickgpt.data.brick_structure import BrickStructure
    b = parse(r["text"])[0]
    try:
        s = gurobi_scores(BrickStructure(b, world_dim=20), threads=1, time_limit=60)
    except Exception as e:
        return {"i": r["i"], "set": r["set"], "err": repr(e)[:200]}
    pb = [float(s[k.x:k.x + k.h, k.y:k.y + k.w, k.z].max()) for k in b]
    return {"i": r["i"], "set": r["set"], "cat": r["cat"], "brick_max": pb, "err": None}


def rerun(workers, per_cat=200):
    from multiprocessing import Pool
    R = [r for r in rows() if r["unstable"] == 1]
    by = defaultdict(list)
    for r in R:
        by[r["cat"]].append(r)
    pick = []
    for c, L in sorted(by.items()):
        L = sorted(L, key=lambda r: (r["set"], r["i"]))
        pick += L if c == GUITAR else random.Random(c).sample(L, min(per_cat, len(L)))
    out = D / "diag_scores.jsonl"
    done = set()
    if out.exists():
        done = {(j["set"], j["i"]) for j in map(json.loads, out.open())}
    todo = [r for r in pick if (r["set"], r["i"]) not in done]
    print(len(pick), "picked", len(todo), "to do", flush=True)
    with out.open("a") as f, Pool(workers) as mp:
        for k, j in enumerate(mp.imap_unordered(_rerun, todo, chunksize=4), 1):
            f.write(json.dumps(j) + "\n")
            if k % 200 == 0:
                f.flush(); print(k, flush=True)


def ftype(pb):
    """candidate-only / candidate+others / prefix-only (candidate fine, others pushed over)."""
    bad = [k for k, v in enumerate(pb) if v >= 1]
    n = len(pb)
    cand = (n - 1) in bad
    if cand and len(bad) == 1:
        return "cand_only"
    if cand:
        return "cand_plus"
    return "prefix_only" if bad else "none"


# ---------- report ----------
def report():
    import torch
    from .gnn import Screener, parse_text, brick_graph, collate
    from .conformal import threshold
    from .run import _u, scores_of
    OUT.mkdir(parents=True, exist_ok=True)
    R = rows()
    for r in R:
        r["b"] = parse_text(r["text"]); r.update(phys(r["b"])); r["g"] = brick_graph(r["b"])
    model = Screener(64, 4); model.load_state_dict(torch.load("results/screen/screener_s0.pt", map_location="cpu"))
    torch.set_num_threads(4)
    sc = scores_of(model, R)
    for r, s in zip(R, sc):
        r["s"] = float(s)
        # score stats only on samples the model never trained on: ID test split + held-out categories
        r["oos"] = r["set"] == "cat" or (r["set"] == "id" and _u(r["sid"]) >= .9)
    cal = [r for r in R if r["set"] == "id" and .8 <= _u(r["sid"]) < .9]
    tau = threshold([r["s"] for r in cal], [r["unstable"] for r in cal], 0.05)
    lg = lambda v: np.log(np.clip(v, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(v, 1e-7, 1 - 1e-7))  # logit; tau ~ 0.998
    ft = {}
    p = D / "diag_scores.jsonl"
    if p.exists():
        for j in map(json.loads, p.open()):
            if not j["err"]:
                ft[(j["set"], j["i"])] = ftype(j["brick_max"])
    group = lambda r: "guitar" if r["cat"] == GUITAR else r["cat"]
    by = defaultdict(list)
    for r in R:
        by[group(r)].append(r)
    keys = ["n", "height", "aspect", "base_area", "cand_overhang", "cand_z", "mean_overhang",
            "frac_overhang_gt_half", "com_offset", "moment"]
    tab = {}
    for c, L in sorted(by.items(), key=lambda t: -len(t[1])):
        O = [r for r in L if r["oos"]]
        U = [r for r in O if r["unstable"]]
        UA = [r for r in L if r["unstable"]]
        miss = [r for r in U if r["s"] < tau]
        ftc = defaultdict(int)
        for r in UA:
            if (r["set"], r["i"]) in ft:
                ftc[ft[(r["set"], r["i"])]] += 1
        nft = sum(ftc.values())
        s_all = np.array([r["s"] for r in O]) if O else np.array([.5])
        tab[c] = {"n": len(L), "n_oos": len(O), "unstable_frac": len(UA) / len(L), "miss@0.05": len(miss) / max(1, len(U)),
                  "n_unstable": len(U),
                  "all_" + "median": {k: float(np.median([r[k] for r in L])) for k in keys},
                  "unstable_median": {k: float(np.median([r[k] for r in UA])) for k in keys} if UA else {},
                  "missed_median": {k: float(np.median([r[k] for r in miss])) for k in keys} if miss else {},
                  "fail_types": {k: v / nft for k, v in ftc.items()} if nft else {}, "n_fail_typed": nft,
                  "unstable_near_tau(<2tau)": float(np.mean([abs(lg(r["s"]) - lg(tau)) < 2 for r in U])) if U else None,
                  "unstable_score_q10_q50": [float(np.quantile([r["s"] for r in U], q)) for q in (.1, .5)] if U else None,
                  "all_frac_s_lt_tau": float((s_all < tau).mean()),
                  "all_frac_ambiguous(tau..0.5)": float((np.abs(lg(s_all) - lg(tau)) < 2).mean()),
                  "all_mean_entropy": float(np.mean(-(s_all * np.log(s_all + 1e-9) + (1 - s_all) * np.log(1 - s_all + 1e-9))))}
        tab[c]["missed_kind_perturb"] = float(np.mean([r["kind"] == "perturb" for r in miss])) if miss else None
    (OUT / "diag.json").write_text(json.dumps({"tau@0.05_seed0": tau, "by_cat": tab}, indent=1))
    lines = [f"tau(α=0.05, seed0) = {tau:.4g}", "",
             "| cat | n | unst | miss | n | height | aspect | cand_oh | com_off | moment | near_tau | ambig | entropy | cand_only/plus/prefix |",
             "|" + "---|" * 14]
    for c, t in tab.items():
        m = t["all_median"]; f = t["fail_types"]
        lines.append(f"| {c} | {t['n']} | {t['unstable_frac']:.3f} | {t['miss@0.05']:.3f} | {m['n']:.0f} | {m['height']:.0f} | "
                     f"{m['aspect']:.2f} | {m['cand_overhang']:.2f} | {m['com_offset']:.2f} | {m['moment']:.2f} | "
                     f"{t['unstable_near_tau(<2tau)'] if t['unstable_near_tau(<2tau)'] is not None else float('nan'):.2f} | "
                     f"{t['all_frac_ambiguous(tau..0.5)']:.3f} | {t['all_mean_entropy']:.3f} | "
                     f"{f.get('cand_only', 0):.2f}/{f.get('cand_plus', 0):.2f}/{f.get('prefix_only', 0):.2f} |")
    (OUT / "diag.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    g = tab["guitar"]
    print("guitar missed medians", json.dumps(g["missed_median"]), "\nguitar unstable medians", json.dumps(g["unstable_median"]),
          "\nmissed perturb frac", g["missed_kind_perturb"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rerun(a.workers) if a.cmd == "rerun" else report()
