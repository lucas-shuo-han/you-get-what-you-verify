"""§4g screener: brick graph + small relational message-passing GNN (plain PyTorch, CPU-friendly).

Graph: one node per brick. Node features: h/8, w/8, x/20, y/20, z/20, [z==0], [is candidate (last brick)], area/16.
Edges (both directions, typed): 0 = j sits on i (z_j = z_i + 1, footprints overlap), 1 = reverse (i on j),
2 = lateral contact (same z, footprints share a side segment). Edge feature: contact size / 16
(stud overlap area for vertical, shared side length for lateral).
"""
from __future__ import annotations

import re
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

# Training/eval path is solver-free: no brickgpt / gurobipy / scipy imports (Kaggle needs only torch + numpy).
SBrick = namedtuple("SBrick", "h w x y z")
_LINE = re.compile(r"^\s*(\d+)x(\d+) \((\d+),(\d+),(\d+)\)\s*$")


def parse_text(text):
    """'HxW (x,y,z)' per line (labels.py output, always well-formed) -> list[SBrick]; raises on any bad line."""
    out = []
    for l in text.strip().splitlines():
        m = _LINE.match(l)
        if not m:
            raise ValueError(f"bad brick line: {l!r}")
        out.append(SBrick(*map(int, m.groups())))
    return out

N_FEAT, N_REL = 8, 3


def brick_graph(bricks):
    """bricks: list with .h .w .x .y .z (candidate last) -> (feat [n,8] float32, edges [E,3] int64 (src,dst,rel), ew [E])."""
    a = np.array([[b.h, b.w, b.x, b.y, b.z] for b in bricks], dtype=np.float32)
    h, w, x, y, z = a.T
    n = len(a)
    feat = np.stack([h / 8, w / 8, x / 20, y / 20, z / 20, (z == 0).astype(np.float32),
                     (np.arange(n) == n - 1).astype(np.float32), h * w / 16], 1).astype(np.float32)
    x1, y1 = x + h, y + w
    ox = np.minimum(x1[:, None], x1[None]) - np.maximum(x[:, None], x[None])  # overlap length along x
    oy = np.minimum(y1[:, None], y1[None]) - np.maximum(y[:, None], y[None])
    dz = z[None] - z[:, None]  # dz[i, j] = z_j - z_i
    vert = (dz == 1) & (ox > 0) & (oy > 0)  # j on top of i
    lat = (dz == 0) & (((ox == 0) & (oy > 0)) | ((oy == 0) & (ox > 0)))
    np.fill_diagonal(lat, False)
    src, dst, rel, ew = [], [], [], []
    i, j = np.nonzero(vert)
    area = (ox * oy)[i, j] / 16
    src += [i, j]; dst += [j, i]; rel += [np.zeros_like(i), np.ones_like(i)]; ew += [area, area]
    i, j = np.nonzero(lat)
    side = np.maximum(ox, oy)[i, j] / 16
    src.append(i); dst.append(j); rel.append(np.full_like(i, 2)); ew.append(side)
    e = np.stack([np.concatenate(src), np.concatenate(dst), np.concatenate(rel)], 1).astype(np.int64)
    return feat, e, np.concatenate(ew).astype(np.float32)


def collate(graphs):
    """list of (feat, edges, ew) -> batched tensors + graph index per node + index of each graph's candidate node."""
    fs, es, ws, gi, cand, off = [], [], [], [], [], 0
    for g, (f, e, w) in enumerate(graphs):
        fs.append(f); ws.append(w)
        es.append(e + np.array([off, off, 0])); gi.append(np.full(len(f), g)); cand.append(off + len(f) - 1)
        off += len(f)
    t = lambda v, d: torch.from_numpy(np.concatenate(v)).to(d)
    return (t(fs, torch.float32), t(es, torch.int64).reshape(-1, 3), t(ws, torch.float32),
            t(gi, torch.int64), torch.tensor(cand), len(graphs))


class Layer(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.msg = nn.ModuleList([nn.Linear(H + 1, H) for _ in range(N_REL)])
        self.upd = nn.Sequential(nn.Linear(H * (N_REL + 1), H), nn.GELU(), nn.Linear(H, H))
        self.ln = nn.LayerNorm(H)

    def forward(self, h, e, ew):
        aggs = []
        for r in range(N_REL):
            m = e[:, 2] == r
            s, d = e[m, 0], e[m, 1]
            msg = torch.relu(self.msg[r](torch.cat([h[s], ew[m, None]], 1)))
            agg = torch.zeros_like(h).index_add_(0, d, msg)
            deg = torch.zeros(len(h), device=h.device).index_add_(0, d, torch.ones_like(d, dtype=h.dtype))
            aggs.append(agg / deg.clamp(min=1)[:, None])
        return self.ln(h + self.upd(torch.cat([h] + aggs, 1)))


class Screener(nn.Module):
    """-> logit of P(unstable)."""
    def __init__(self, H=64, L=4):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(N_FEAT, H), nn.GELU(), nn.Linear(H, H))
        self.layers = nn.ModuleList([Layer(H) for _ in range(L)])
        self.head = nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Linear(H, 1))

    def forward(self, feat, e, ew, gi, cand, G):
        h = self.emb(feat)
        for l in self.layers:
            h = l(h, e, ew)
        H = h.shape[1]
        cnt = torch.zeros(G, device=h.device).index_add_(0, gi, torch.ones_like(gi, dtype=h.dtype))
        mean = torch.zeros(G, H, device=h.device).index_add_(0, gi, h) / cnt[:, None]
        mx = torch.full((G, H), -1e9, device=h.device).scatter_reduce(0, gi[:, None].expand(-1, H), h, "amax")
        return self.head(torch.cat([mean, mx, h[cand]], 1)).squeeze(1)
