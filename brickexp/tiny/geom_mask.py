"""Decode-time geometry masking (DIRECTION §4a): forbid tokens that would make the current brick collide with
occupied cells or leave the 20^3 grid. Nothing about floating / stability is masked.

Per-row tracker mirrors vocab.decode (BACK pops, FB skips 4 tokens, BOS/CAT ignored). Inside a brick the grammar
T -> X -> Y -> Z is enforced (only the next coordinate kind, and only values that keep >= 1 valid completion), so a
masked brick always decodes and never collides / goes out of bounds. At a brick boundary only infeasible types are
masked; EOS, BACK, FB etc. stay allowed.
"""
from __future__ import annotations

import numpy as np
import torch

from . import vocab as V

G = V.GRID
TYPES = [tuple(map(int, t.split("x"))) for t in V.BRICK_TYPES]
T_IDS = [V.ID[f"T_{t}"] for t in V.BRICK_TYPES]
X_IDS = np.array([V.ID[f"X{i}"] for i in range(G)])
Y_IDS = np.array([V.ID[f"Y{i}"] for i in range(G)])
Z_IDS = np.array([V.ID[f"Z{i}"] for i in range(G)])
TOK = V.TOKENS


class GeomTracker:
    def __init__(self, prefix=()):
        self.occ = np.zeros((G, G, G), dtype=np.int32)  # occupancy counts (prefix may contain a colliding bad brick)
        self.bricks = []
        self.pending = []  # [type_idx, x, y] collected so far for the current brick
        self.skip = 0
        self._valid = None  # cache: per type, bool [x, y, z] placements that fit and do not collide
        for t in prefix:
            self.update(int(t), strict=False)

    def _valid_all(self):
        if self._valid is None:
            free = (self.occ == 0).astype(np.int32)
            # 2D prefix sums per z over x,y of occupied cells
            occ = 1 - free
            cs = np.zeros((G + 1, G + 1, G), dtype=np.int32)
            cs[1:, 1:] = occ.cumsum(0).cumsum(1)
            self._valid = []
            for h, w in TYPES:
                v = np.zeros((G, G, G), dtype=bool)
                nx, ny = G - h + 1, G - w + 1
                s = cs[h:h + nx, w:w + ny] - cs[:nx, w:w + ny] - cs[h:h + nx, :ny] + cs[:nx, :ny]
                v[:nx, :ny] = s == 0
                self._valid.append(v)
        return self._valid

    def allowed(self) -> np.ndarray | None:
        """Bool mask over the vocab, or None (no restriction: skipping feedback tokens)."""
        if self.skip:
            return None
        m = np.ones(V.VOCAB_SIZE, dtype=bool)
        va = self._valid_all()
        if not self.pending:
            for ti, tid in enumerate(T_IDS):
                if not va[ti].any():
                    m[tid] = False
            return m
        m[:] = False
        v = va[self.pending[0]]
        if len(self.pending) == 1:
            m[X_IDS[v.any(axis=(1, 2))]] = True
        elif len(self.pending) == 2:
            m[Y_IDS[v[self.pending[1]].any(axis=1)]] = True
        else:
            m[Z_IDS[v[self.pending[1], self.pending[2]]]] = True
        return m

    def update(self, t: int, strict=True):
        if self.skip:
            self.skip -= 1
            return
        name = TOK[t] if 0 <= t < V.VOCAB_SIZE else "?"
        if self.pending:
            want = "XYZ"[len(self.pending) - 1]
            if name[:1] == want and name[1:].isdigit():
                self.pending.append(int(name[1:]))
                if len(self.pending) == 4:
                    ti, x, y, z = self.pending
                    h, w = TYPES[ti]
                    self.occ[x:x + h, y:y + w, z] += 1
                    self.bricks.append((ti, x, y, z))
                    self.pending = []
                    self._valid = None
                return
            assert not strict, f"grammar violated under mask: {name}"
            self.pending = []  # decode treats the T as a bad token and re-reads from here
        if name == "BACK":
            if self.bricks:
                ti, x, y, z = self.bricks.pop()
                h, w = TYPES[ti]
                self.occ[x:x + h, y:y + w, z] -= 1
                self._valid = None
        elif name == "FB":
            self.skip = 4
        elif name.startswith("T_"):
            self.pending = [T_IDS.index(t)]


def apply(last: torch.Tensor, trackers) -> torch.Tensor:
    """last: [n, vocab] logits for these trackers' rows -> masked copy."""
    ms = [tr.allowed() for tr in trackers]
    if all(m is None for m in ms):
        return last
    mask = torch.from_numpy(np.stack([np.ones(V.VOCAB_SIZE, bool) if m is None else m for m in ms])).to(last.device)
    return last.masked_fill(~mask, -1e9)
