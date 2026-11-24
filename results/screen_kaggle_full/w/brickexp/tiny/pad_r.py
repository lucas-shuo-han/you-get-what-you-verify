"""DIRECTION §4d: token-match the R arm to R+F by appending extra event-bearing structures (R format: bad brick,
BACK, correct brick) from the unused pool, the way gen_data pads S. Only R.npy changes (old copy -> R_v1.npy).

The pool is rebuilt exactly as gen_data.main does (same filter, sort, shuffle), and starts after the
s_extra_structs entries that S already used. Events use the same per-structure RNG and the same settings
(p_err, max_err, unstable) recorded in stats.json.

Usage: python -m brickexp.tiny.pad_r --data data/tiny/n4000_s0 [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

from .gen_data import _events_worker, category_map, encode
from ..checker import parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64, help="pool structures per event-mining round")
    a = ap.parse_args()
    from ..data import load_split

    d = Path(a.data)
    st = json.loads((d / "stats.json").read_text())
    g = st["args"]
    if (d / "R_v1.npy").exists():
        r_seqs = list(np.load(d / "R_v1.npy", allow_pickle=True))  # idempotent: always pad the v1 R
    else:
        r_seqs = list(np.load(d / "R.npy", allow_pickle=True))
        shutil.copy(d / "R.npy", d / "R_v1.npy")
    target = int(st["tokens"]["RF"])

    recs = [r for r in load_split("train") if r["n_bricks"] <= g["max_bricks"]]
    cmap = category_map(load_split("train"))
    rng = random.Random(g["seed"])
    recs = sorted(recs, key=lambda r: r["structure_id"])
    rng.shuffle(recs)
    pool = recs[g["n_struct"]:][st["s_extra_structs"]:]

    n_tok, extra, n_ev, pos = int(sum(map(len, r_seqs))), 0, 0, 0
    while n_tok < target:
        chunk = pool[pos:pos + a.batch]
        assert chunk, "pool exhausted"
        pos += len(chunk)
        jobs = [(r["structure_id"], r["bricks"], g["seed"], g["p_err"], g["max_err"], g["unstable"]) for r in chunk]
        if a.workers > 1:
            from multiprocessing import Pool
            with Pool(a.workers) as mp:
                res = mp.map(_events_worker, jobs, chunksize=4)
        else:
            res = [_events_worker(j) for j in jobs]
        for r, (evs, _) in zip(chunk, res):
            if n_tok >= target:
                break
            if not evs:  # event-bearing structures only
                continue
            seq = encode(cmap[r["category_id"]], parse(r["bricks"])[0], evs, "none")
            r_seqs.append(np.array(seq, dtype=np.uint8)); n_tok += len(seq); extra += 1; n_ev += len(evs)

    np.save(d / "R.npy", np.array(r_seqs, dtype=object), allow_pickle=True)
    st["tokens"]["R"] = n_tok
    st["max_len"]["R"] = int(max(map(len, r_seqs)))
    st["r_pad"] = {"r_extra_structs": extra, "r_extra_events": n_ev, "pool_scanned": pos,
                   "pool_offset": st["s_extra_structs"], "r_tokens_v1": int(sum(map(len, np.load(d / "R_v1.npy", allow_pickle=True)))),
                   "ratio_R_over_RF": n_tok / target}
    (d / "stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st["r_pad"] | {"R_tokens": n_tok, "RF_tokens": target}, indent=1))


if __name__ == "__main__":
    main()
