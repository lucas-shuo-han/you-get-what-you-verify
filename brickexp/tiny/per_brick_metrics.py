"""Per-brick metrics (DIRECTION §4b), recomputed from saved generations; no model rerun.

Input: the "text" field of eval*.jsonl = the RESOLVED brick sequence (after BACK retractions; raw token streams are
not saved). Definition (incremental, in output order): brick i is floating iff, in the structure made of bricks
0..i only, it is not at z=0 and has nothing directly below or above it (checker's floating rule applied to the
prefix). So a brick placed in mid-air counts as floating even if a later brick would support it.
For each prompt's attempt k (default 0):
  n_placed    = number of bricks
  n_floating  = number of floating bricks
  first_error = number of bricks before the first floating one (= n_placed if none)
Aggregates with per-prompt bootstrap 95% CI:
  floating_frac = sum(n_floating) / sum(n_placed)   (ratio of sums, resampling prompts)
  first_error   = mean over prompts
§4c (--post_filter, PRIMARY): read unmasked eval.jsonl; scan resolved bricks in generation order, drop unparseable,
out-of-bounds, and bricks colliding with an already-kept brick (earlier wins); on kept bricks, --support rule:
  mixed (default): supported iff z=0, or an EARLIER kept brick directly below, or ANY kept brick directly above
  incremental: only earlier kept bricks;   final: any kept brick (checker rule)
Also reports filtered_out_frac = dropped / generated (bookkeeping).
Usage: python -m brickexp.tiny.per_brick_metrics RUN_DIR_OR_JSONL [...] [--k 0] [--boot 10000] [--out metrics.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import _VENDOR  # noqa: F401
from ..checker import parse


def brick_stats(text: str, world_dim: int = 20) -> dict:
    """-> {n_placed, n_floating, first_error, floating_idx}; empty/unparseable text -> zeros."""
    from brickgpt.data.brick_structure import BrickStructure
    bricks = parse(text, world_dim)[0] if text else []
    bricks = bricks or []
    s = BrickStructure([], world_dim=world_dim)
    fl = []
    for i, b in enumerate(bricks):  # incremental: support only from bricks 0..i
        s.add_brick(b)
        if s.brick_floats(b):
            fl.append(i)
    n = len(bricks)
    return {"n_placed": n, "n_floating": len(fl), "first_error": fl[0] if fl else n, "floating_idx": fl}


SUPPORT_RULES = ("mixed", "incremental", "final")


def post_filter(bricks, world_dim: int = 20):
    """§4c: scan in generation order; drop out-of-bounds bricks and bricks colliding with an already-kept brick
    (earlier kept wins). -> (kept, n_dropped_oob, n_dropped_collision)."""
    from brickgpt.data.brick_structure import BrickStructure
    s = BrickStructure([], world_dim=world_dim)
    kept, oob, col = [], 0, 0
    for b in bricks:
        if not (b.x >= 0 and b.y >= 0 and b.x + b.h <= world_dim and b.y + b.w <= world_dim and 0 <= b.z < world_dim):
            oob += 1
        elif s.brick_collides(b):
            col += 1
        else:
            s.add_brick(b); kept.append(b)
    return kept, oob, col


def floating_flags(bricks, rule: str = "mixed", world_dim: int = 20) -> list[bool]:
    """Per-brick floating flags on a collision-free in-bounds sequence.
    mixed (§4c primary): supported iff z=0, or an EARLIER brick directly below, or ANY brick directly above.
    incremental: only earlier bricks count (below or above).   final: any brick (checker rule on the whole structure)."""
    import numpy as np
    G = world_dim
    first = np.full((G, G, G), -1, dtype=np.int64)  # index of the brick occupying each cell
    for i, b in enumerate(bricks):
        first[b.x:b.x + b.h, b.y:b.y + b.w, b.z] = i
    out = []
    for i, b in enumerate(bricks):
        if b.z == 0:
            out.append(False); continue
        below = first[b.x:b.x + b.h, b.y:b.y + b.w, b.z - 1]
        above = first[b.x:b.x + b.h, b.y:b.y + b.w, b.z + 1] if b.z + 1 < G else np.full(1, -1)
        if rule == "mixed":
            sup = ((below >= 0) & (below < i)).any() or (above >= 0).any()
        elif rule == "incremental":
            sup = ((below >= 0) & (below < i)).any() or ((above >= 0) & (above < i)).any()
        elif rule == "final":
            sup = (below >= 0).any() or (above >= 0).any()
        else:
            raise ValueError(rule)
        out.append(not sup)
    return out


def post_filter_stats(text: str, rule: str = "mixed", world_dim: int = 20) -> dict:
    """§4c per-prompt stats: drop unparseable lines, then post_filter, then floating flags on kept bricks."""
    from brickgpt.data.brick_structure import Brick
    import re
    pat = re.compile(r"^\s*(\d+)x(\d+) \((-?\d+),(-?\d+),(-?\d+)\)\s*$")
    bricks, n_unparse = [], 0
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        m = pat.match(line)
        if m:
            h, w, x, y, z = map(int, m.groups())
            bricks.append(Brick(h=h, w=w, x=x, y=y, z=z))
        else:
            n_unparse += 1
    kept, oob, col = post_filter(bricks, world_dim)
    fl = floating_flags(kept, rule, world_dim)
    idx = [i for i, f in enumerate(fl) if f]
    return {"n_placed": len(kept), "n_floating": len(idx), "first_error": idx[0] if idx else len(kept),
            "n_generated": len(bricks) + n_unparse, "n_dropped": n_unparse + oob + col,
            "n_dropped_oob": oob, "n_dropped_collision": col, "n_unparseable": n_unparse}


def bootstrap(rows: list[dict], n_boot: int = 10000, seed: int = 0) -> dict:
    P = len(rows)
    nf = np.array([r["n_floating"] for r in rows], float)
    npl = np.array([r["n_placed"] for r in rows], float)
    fe = np.array([r["first_error"] for r in rows], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, P, size=(n_boot, P))
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_b = nf[idx].sum(1) / npl[idx].sum(1)
    fe_b = fe[idx].mean(1)
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return {"n_prompts": P, "n_empty": int((npl == 0).sum()),
            "floating_frac": float(nf.sum() / max(npl.sum(), 1)), "floating_frac_ci95": ci(frac_b),
            "first_error": float(fe.mean()), "first_error_ci95": ci(fe_b),
            "mean_bricks": float(npl.mean()),
            **({"filtered_out_frac": float(sum(r["n_dropped"] for r in rows) / max(1, sum(r["n_generated"] for r in rows))),
                "filtered_out_frac_ci95": ci(np.array([r["n_dropped"] for r in rows], float)[idx].sum(1)
                                             / np.maximum(1, np.array([r["n_generated"] for r in rows], float)[idx].sum(1)))}
               if rows and "n_dropped" in rows[0] else {})}


def score_file(path, k: int = 0, n_boot: int = 10000, post_filter_rule: str | None = None) -> dict:
    """post_filter_rule None: legacy §4b scoring (masked runs). Else §4c post-filter with that support rule."""
    path = Path(path)
    if path.is_dir():
        path = path / ("eval.jsonl" if post_filter_rule else "eval_geom.jsonl")
    rows = []
    for line in open(path, encoding="utf-8"):
        s = json.loads(line)
        att = [a for a in s["attempts"] if a["k"] == k]
        if post_filter_rule:
            rows.append(post_filter_stats(att[0]["text"] if att else "", post_filter_rule))
        else:
            rows.append(brick_stats(att[0]["text"]) if att else {"n_placed": 0, "n_floating": 0, "first_error": 0})
    return {"file": str(path), "k": k, "mode": f"post_filter:{post_filter_rule}" if post_filter_rule else "legacy",
            **bootstrap(rows, n_boot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--post_filter", action="store_true", help="§4c primary: unmasked outputs, post-hoc geometry filter")
    ap.add_argument("--support", default="mixed", choices=SUPPORT_RULES, help="floating rule for --post_filter")
    a = ap.parse_args()
    res = [score_file(p, a.k, a.boot, a.support if a.post_filter else None) for p in a.paths]
    for r in res:
        print(json.dumps(r))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
