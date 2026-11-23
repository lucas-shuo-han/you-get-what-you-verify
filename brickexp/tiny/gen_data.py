"""Build token datasets for the four arms (DIRECTION v2 §3).

Source: stable StableText2Brick train structures (dataset brick order).
Error events (Physics-of-LMs-style local retries): before true brick i, insert a perturbed brick c that FAILS
the checker against the prefix b_1..b_{i-1}; then feedback (arm-dependent), then BACK, then b_i.
  S    : success only; padded with extra structures until its token count matches R+F.
  R+F  : events with true feedback (failure type + index of the bad brick).
  R+F~ : same events/sequences, feedback shuffled across events (same format and length, wrong content).
  R    : same events, no feedback (bad brick, BACK).
Failure labels for cheap types (out_of_bounds/collision/floating) need no solver; --unstable also searches for
candidates that pass the cheap checks but fail Gurobi (slow; cached by text).

Usage: python -m brickexp.tiny.gen_data --n_struct 4000 --seed 0 --out data/tiny/n4000_s0
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from . import vocab as V
from ..checker import parse

PERTURB_TRIES = 30


def category_map(recs) -> dict[str, int]:
    cats = sorted({r["category_id"] for r in recs})
    assert len(cats) <= V.N_CAT, len(cats)
    return {c: i for i, c in enumerate(cats)}


def cheap_fail(prefix_struct, c, world_dim=20) -> str | None:
    """Failure type of adding brick c to a valid prefix (checker order), or None if it passes cheap checks."""
    if not (c.x >= 0 and c.y >= 0 and c.x + c.h <= world_dim and c.y + c.w <= world_dim and 0 <= c.z < world_dim):
        return "out_of_bounds"
    if prefix_struct.brick_collides(c):
        return "collision"
    prefix_struct.add_brick(c)
    try:
        if prefix_struct.brick_floats(c):
            return "floating"
    finally:
        prefix_struct.undo_add_brick()
    return None


def perturb(b, rng: random.Random):
    from brickgpt.data.brick_structure import Brick
    kind = rng.random()
    h, w, x, y, z = b.h, b.w, b.x, b.y, b.z
    if kind < 0.4:  # lateral shift
        x += rng.choice([-3, -2, -1, 1, 2, 3]); y += rng.choice([-3, -2, -1, 0, 1, 2, 3])
    elif kind < 0.7:  # vertical shift (often floating / collision)
        z += rng.choice([-1, 1, 2, 3])
    else:  # different brick type at a nearby spot
        h, w = map(int, rng.choice(V.BRICK_TYPES).split("x"))
        x += rng.choice([-1, 0, 1]); y += rng.choice([-1, 0, 1])
    x, y, z = max(0, min(19, x)), max(0, min(19, y)), max(0, min(19, z))
    return Brick(h=h, w=w, x=x, y=y, z=z)


def make_events(bricks, rng: random.Random, p_err: float, max_err: int, unstable: bool, gurobi_cache: dict):
    """-> list of (i, bad_brick, fail_type, index1): error inserted before true brick i (0-based)."""
    from brickgpt.data.brick_structure import BrickStructure
    s = BrickStructure([], world_dim=20)
    events = []
    for i, b in enumerate(bricks):
        if len(events) < max_err and i > 0 and rng.random() < p_err:
            for _ in range(PERTURB_TRIES):
                c = perturb(b, rng)
                f = cheap_fail(s, c)
                if f is None and unstable:
                    f = _gurobi_fail(s, c, gurobi_cache)
                if f is not None:
                    events.append((i, c, f, i + 1))  # bad brick would be brick number i+1
                    break
        s.add_brick(b)
    return events


def _events_worker(job):
    """One structure -> events, with its own deterministic RNG (so parallel == serial output)."""
    sid, text, seed, p_err, max_err, unstable = job
    bricks, err, _ = parse(text)
    assert bricks is not None, err
    rng = random.Random(f"{seed}-{sid}")
    cache = {}
    evs = make_events(bricks, rng, p_err, max_err, unstable, cache)
    return evs, len(cache)


def _gurobi_fail(s, c, cache):
    from ..checker import gurobi_scores
    from brickgpt.data.brick_structure import BrickStructure
    t = V.bricks_to_text(s.bricks + [c])
    if t not in cache:
        st = BrickStructure(list(s.bricks) + [c], world_dim=20)
        try:
            cache[t] = bool((gurobi_scores(st, threads=1, time_limit=30) >= 1).any())
        except Exception:
            cache[t] = None  # solver error: do not use as a label
    return "unstable" if cache[t] else None


def encode(cat: int, bricks, events, fb_mode: str, fb_override=None) -> list[int]:
    """fb_mode: 'true' | 'shuffled' (use fb_override list aligned with events) | 'none'."""
    ev = {e[0]: (k, e) for k, e in enumerate(events)}
    ids = [V.BOS, V.ID[f"CAT_{cat}"]]
    for i, b in enumerate(bricks):
        if i in ev:
            k, (_, c, f, idx) = ev[i]
            ids += V.brick_tokens(c)
            if fb_mode == "true":
                ids += V.feedback_tokens(f, idx)
            elif fb_mode == "shuffled":
                ids += V.feedback_tokens(*fb_override[k])
            ids.append(V.BACK)
        ids += V.brick_tokens(b)
    ids.append(V.EOS)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_struct", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_err", type=float, default=0.1, help="per-brick probability of an error event")
    ap.add_argument("--max_err", type=int, default=8)
    ap.add_argument("--max_bricks", type=int, default=300)
    ap.add_argument("--unstable", action="store_true", help="also mine Gurobi-unstable bad bricks (slow)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from ..data import load_split
    t0 = time.time()
    recs = [r for r in load_split("train") if r["n_bricks"] <= a.max_bricks]
    cmap = category_map(load_split("train"))
    rng = random.Random(a.seed)
    recs = sorted(recs, key=lambda r: r["structure_id"])
    rng.shuffle(recs)
    main_recs, pool = recs[:a.n_struct], recs[a.n_struct:]

    jobs = [(r["structure_id"], r["bricks"], a.seed, a.p_err, a.max_err, a.unstable) for r in main_recs]
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as mp_pool:
            results = mp_pool.map(_events_worker, jobs, chunksize=8)
    else:
        results = [_events_worker(j) for j in jobs]
    n_gurobi = sum(n for _, n in results)
    rows = [(cmap[r["category_id"]], parse(r["bricks"])[0], evs) for r, (evs, _) in zip(main_recs, results)]
    all_fb = [(f, idx) for _, _, evs in rows for (_, _, f, idx) in evs]
    perm = list(range(len(all_fb)))
    rng.shuffle(perm)
    shuffled = [all_fb[j] for j in perm]
    n_same = sum(all_fb[k] == shuffled[k] for k in range(len(all_fb)))

    arms = {"RF": [], "RFs": [], "R": []}
    k = 0
    for cat, bricks, evs in rows:
        arms["RF"].append(encode(cat, bricks, evs, "true"))
        arms["RFs"].append(encode(cat, bricks, evs, "shuffled", shuffled[k:k + len(evs)]))
        arms["R"].append(encode(cat, bricks, evs, "none"))
        k += len(evs)
    target = sum(map(len, arms["RF"]))
    arms["S"] = [encode(cat, bricks, [], "none") for cat, bricks, _ in rows]
    n_tok, n_extra = sum(map(len, arms["S"])), 0
    for r in pool:  # pad S with extra stable structures to match R+F tokens
        if n_tok >= target:
            break
        bricks, _, _ = parse(r["bricks"])
        seq = encode(cmap[r["category_id"]], bricks, [], "none")
        arms["S"].append(seq); n_tok += len(seq); n_extra += 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"args": vars(a), "n_struct": len(rows), "n_events": len(all_fb), "s_extra_structs": n_extra, "n_gurobi_solves": n_gurobi,
             "shuffled_feedback_identical_frac": n_same / max(1, len(all_fb)),
             "fail_types": {f: sum(1 for x, _ in all_fb if x == f) for f in V.FAIL_TYPES},
             "tokens": {k: int(sum(map(len, v))) for k, v in arms.items()},
             "max_len": {k: int(max(map(len, v))) for k, v in arms.items()},
             "category_map": cmap, "seconds": time.time() - t0}
    for name, seqs in arms.items():
        np.save(out / f"{name}.npy", np.array([np.array(s, dtype=np.uint8) for s in seqs], dtype=object),
                allow_pickle=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "category_map"}, indent=1))


if __name__ == "__main__":
    main()

