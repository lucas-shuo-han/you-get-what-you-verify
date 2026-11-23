"""§4g labels: (prefix + one candidate brick) -> Gurobi stable/unstable.

Each sample i of a set is deterministic (RNG seeded by set name and i): pick a train structure from the set's pool,
cut a random prefix of L bricks, add a candidate (70% perturbed next brick, 30% the true next brick) that passes the
cheap checks (bounds, collision, floating), then label with Gurobi (unstable iff any voxel score >= 1).
Solver errors are recorded with label None (dropped downstream, counted).

Sets: id   = non-held-out categories, structure size L+1 <= 100
      cat  = the held-out categories (HELD), size <= 100
      size = non-held-out categories, size 150..300
Output is append-only JSONL keyed by i, so a rerun resumes.

Usage: python -m brickexp.screen.labels --set id --n 40000 --out data/screen/id.jsonl --workers 16
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

HELD = ["03467517", "02876657", "03593526"]  # guitar, bottle, vase: mid-size categories (1361/854/732 train structs)
SIZES = {"id": (4, 100), "cat": (4, 100), "size": (150, 300), "loco": (4, 100)}
# §4i top-up for small ID categories (target ~150 unstable each; n from the observed unstable rate, x1.15)
LOCO_N = [("04468005", 810), ("02871439", 1230), ("03991062", 360), ("03797390", 620), ("03928116", 380),
          ("02818832", 425), ("02880940", 125), ("04460130", 2090), ("02942699", 430), ("02801938", 1460),
          ("02843684", 1730)]


def loco_cat(i):
    for c, n in LOCO_N:
        if i < n:
            return c
        i -= n
    raise IndexError("loco index out of range")
P_PERTURB = 0.7
_POOL = None


def pool(set_name):
    from ..data import load_split
    lo, hi = SIZES[set_name]
    recs = sorted(load_split("train"), key=lambda r: r["structure_id"])
    held = set_name == "cat"  # loco: same pool as id, filtered per sample in make_sample
    return [(r["structure_id"], r["category_id"], r["bricks"]) for r in recs
            if (r["category_id"] in HELD) == held and r["n_bricks"] >= lo]


def make_sample(set_name, i, recs, tries=200):
    """-> (meta, bricks) for sample i; deterministic."""
    from ..checker import parse
    from ..tiny.gen_data import perturb, cheap_fail
    from brickgpt.data.brick_structure import BrickStructure
    lo, hi = SIZES[set_name]
    rng = random.Random(f"screen-{set_name}-{i}")
    if set_name == "loco":
        c = loco_cat(i); recs = [r for r in recs if r[1] == c]
    for _ in range(tries):
        sid, cat, text = rng.choice(recs)
        b = parse(text)[0]
        top = min(len(b) - 1, hi - 1)  # L+1 <= hi and there is a next brick
        if top < lo - 1:
            continue
        L = rng.randint(lo - 1, top)
        s = BrickStructure(list(b[:L]), world_dim=20)
        kind = "perturb" if rng.random() < P_PERTURB else "true"
        c = perturb(b[L], rng) if kind == "perturb" else b[L]
        if cheap_fail(s, c) is not None:
            continue
        return {"i": i, "sid": sid, "cat": cat, "L": L, "kind": kind}, list(b[:L]) + [c]
    raise RuntimeError(f"no sample for {set_name} {i}")


def _init(set_name):
    global _POOL
    _POOL = pool(set_name)


def _job(args):
    set_name, i = args
    from ..checker import gurobi_scores
    from ..tiny import vocab as V
    from brickgpt.data.brick_structure import BrickStructure
    meta, bricks = make_sample(set_name, i, _POOL)
    t = time.time()
    try:
        sc = gurobi_scores(BrickStructure(bricks, world_dim=20), threads=1, time_limit=60)
        lab, err = int((sc >= 1).any()), None
    except Exception as e:  # recorded, never silently relabelled
        lab, err = None, repr(e)[:200]
    meta.update(text=V.bricks_to_text(bricks), unstable=lab, err=err, sec=round(time.time() - t, 3))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SIZES))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["i"] for l in out.open() if l.strip()}
    todo = [(a.set, i) for i in range(a.n) if i not in done]
    print(f"{a.set}: {len(done)} done, {len(todo)} to do", flush=True)
    t0 = time.time()
    with out.open("a") as f:
        if a.workers > 1:
            from multiprocessing import Pool
            with Pool(a.workers, initializer=_init, initargs=(a.set,)) as mp:
                it = mp.imap_unordered(_job, todo, chunksize=4)
                for k, r in enumerate(it, 1):
                    f.write(json.dumps(r) + "\n")
                    if k % 500 == 0:
                        f.flush(); print(f"{k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        else:
            _init(a.set)
            for r in map(_job, todo):
                f.write(json.dumps(r) + "\n")
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
