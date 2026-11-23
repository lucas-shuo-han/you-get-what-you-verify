"""§4j truss generator: 8 families, random geometry / sections / loads; labels from fem.judge.

Families (2D unless noted): pratt, howe, warren, ktruss, fink (pitched, W webs), bowstring, space (3D double-layer
grid), random (Delaunay of random points). Panels: train-size 4..10, size-shift 12..20.
Every sample i is deterministic (RNG seeded by family, size-set and i). A mechanism (cond > 1e12) -> resample.
Global load scale LAMBDA is the only tuned constant; tuned on train-size samples pooled over families (§4j).

Usage: python -m brickexp.truss.gen --n 3000 --size small --out data/truss/small.jsonl
       python -m brickexp.truss.gen --calibrate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .fem import judge

FAMILIES = ["pratt", "howe", "warren", "ktruss", "fink", "bowstring", "space", "random"]
FAMILIES10 = FAMILIES + ["tower", "box"]  # §4k round 2: two more 3D families
THREE_D = ("space", "tower", "box")  # vertical axis = z (index 2)
LAM_FAM = {}  # §4k: per-new-family lambda multiplier, only if pooled unsafe drifts > 10 pp (see DEBUG_LOG)
PANELS = {"small": (4, 10), "large": (12, 20)}
LAMBDA = 0.35  # pooled train-size unsafe ~32% (results/truss/calibration.json, lam 0.3->28%, 0.4->35%)
PIN, ROLL, PLANE = [True, True, True], [False, True, True], [False, False, True]


class B:
    def __init__(self):
        self.X, self.M, self.grp = [], [], []

    def node(self, *p):
        self.X.append([*p] + [0.0] * (3 - len(p))); return len(self.X) - 1

    def bar(self, i, j, g):
        self.M.append([i, j]); self.grp.append(g)


_FLIP = {"p": 0.0, "rng": None}  # §4k round 5: per-panel diagonal flip (mechanism test only)


def _flipped(left):
    return (not left) if _FLIP["p"] and _FLIP["rng"].random() < _FLIP["p"] else left


def _flat(fam, n, S, h, rng):
    b = B(); s = S / n
    bot = [b.node(i * s, 0) for i in range(n + 1)]
    for i in range(n):
        b.bar(bot[i], bot[i + 1], "chord")
    if fam == "warren":
        top = [b.node((i + .5) * s, h) for i in range(n)]
        for i in range(n):
            b.bar(bot[i], top[i], "web"); b.bar(top[i], bot[i + 1], "web")
            if i:
                b.bar(top[i - 1], top[i], "chord")
        return b, bot, top
    if fam == "bowstring":
        ys = [h * 4 * (i * s) * (S - i * s) / S ** 2 for i in range(n + 1)]
        top = [bot[0]] + [b.node(i * s, ys[i]) for i in range(1, n)] + [bot[n]]
    else:
        top = [b.node(i * s, h) for i in range(n + 1)]
    for i in range(n):
        b.bar(top[i], top[i + 1], "chord")
    rng_v = range(1, n) if fam == "bowstring" else range(n + 1)
    if fam == "ktruss":
        mid = {i: b.node(i * s, h / 2) for i in range(1, n)}
        for i in range(n + 1):
            if i in mid:
                b.bar(bot[i], mid[i], "web"); b.bar(mid[i], top[i], "web")
            else:
                b.bar(bot[i], top[i], "web")
        for i in range(n):
            if _flipped(i < n / 2):  # K points to the centre: diagonals from the inner vertical's mid node to the outer one
                j, o = i + 1, i
            else:
                j, o = i, i + 1
            if j in mid:
                b.bar(mid[j], top[o], "web"); b.bar(mid[j], bot[o], "web")
            else:
                b.bar(bot[i], top[i + 1], "web")
        return b, bot, top
    for i in rng_v:
        b.bar(bot[i], top[i], "web")
    for i in range(n):
        left = _flipped(i < n / 2)
        if fam in ("pratt", "bowstring"):  # diagonals slope down toward the centre (tension under gravity)
            p = (top[i], bot[i + 1]) if left else (bot[i], top[i + 1])
        else:  # howe
            p = (bot[i], top[i + 1]) if left else (top[i], bot[i + 1])
        if p[0] != p[1] and not (fam == "bowstring" and (i == 0 or i == n - 1)):
            b.bar(*p, "web")
    return b, bot, top


def _fink(n, S, h, rng):
    n += n % 2
    b = B(); s = S / n
    bot = [b.node(i * s, 0) for i in range(n + 1)]
    for i in range(n):
        b.bar(bot[i], bot[i + 1], "chord")
    y = lambda x: h * (1 - abs(x - S / 2) / (S / 2))
    top = [b.node((i + .5) * s, y((i + .5) * s)) for i in range(n)]
    apex = b.node(S / 2, h)
    chain = [bot[0]] + top[:n // 2] + [apex] + top[n // 2:] + [bot[n]]
    for a, c in zip(chain, chain[1:]):
        b.bar(a, c, "chord")
    for i in range(n):
        b.bar(bot[i], top[i], "web"); b.bar(top[i], bot[i + 1], "web")
    b.bar(apex, bot[n // 2], "web")
    return b, bot, top + [apex]


def _random(n, S, h, rng):
    from scipy.spatial import Delaunay
    b = B()
    k = 2 * (n + 1) + int(rng.integers(0, n + 1))
    P = [(0, 0), (S, 0)] + [(i * S / n, 0) for i in range(1, n)] + [(rng.uniform(0, S), rng.uniform(.25 * h, h)) for _ in range(k - n - 1)]
    P = np.array(P)
    ids = [b.node(*p) for p in P]
    tri = Delaunay(P)
    E = set()
    for t in tri.simplices:
        for a, c in ((0, 1), (1, 2), (0, 2)):
            E.add(tuple(sorted((t[a], t[c]))))
    for a, c in sorted(E):
        b.bar(ids[a], ids[c], "chord" if P[a, 1] < 1e-9 and P[c, 1] < 1e-9 else "web")
    order = np.argsort(P[:, 0])
    bot = [ids[0], ids[1]]
    top = [ids[i] for i in order if P[i, 1] > .5 * h] or [ids[int(np.argmax(P[:, 1]))]]
    return b, bot, top


def _space(n, S, h, rng):
    ny = max(2, int(round(n * rng.uniform(.5, 1))))
    b = B(); s = S / n
    bot = {(i, j): b.node(i * s, j * s, 0) for i in range(n + 1) for j in range(ny + 1)}
    top = {(i, j): b.node((i + .5) * s, (j + .5) * s, h) for i in range(n) for j in range(ny)}
    for (i, j), v in bot.items():
        if (i + 1, j) in bot: b.bar(v, bot[(i + 1, j)], "chord")
        if (i, j + 1) in bot: b.bar(v, bot[(i, j + 1)], "chord")
    for (i, j), v in top.items():
        if (i + 1, j) in top: b.bar(v, top[(i + 1, j)], "chord")
        if (i, j + 1) in top: b.bar(v, top[(i, j + 1)], "chord")
        for di in (0, 1):
            for dj in (0, 1):
                b.bar(v, bot[(i + di, j + dj)], "web")
    corners = [bot[(0, 0)], bot[(n, 0)], bot[(0, ny)], bot[(n, ny)]]
    return b, corners, list(top.values()), s * s


def _tower(n, S, h, rng):
    """square lattice tower, height S (n panels), base width S/sd; legs taper; all faces + top plan triangulated."""
    b = B(); H = S; w0 = h; tp = rng.uniform(0, .5); ph = H / n
    lv = []
    for k in range(n + 1):
        w = w0 * (1 - tp * k / n) / 2
        lv.append([b.node(sx * w, sy * w, k * ph) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    for k in range(n + 1):
        for c in range(4):
            if k: b.bar(lv[k - 1][c], lv[k][c], "chord")
            if k: b.bar(lv[k][c], lv[k][(c + 1) % 4], "web")
            if k: b.bar(lv[k - 1][c] if k % 2 else lv[k - 1][(c + 1) % 4], lv[k][(c + 1) % 4] if k % 2 else lv[k][c], "web")
    b.bar(lv[n][0], lv[n][2], "web")
    return b, lv[0], [v for L in lv[1:] for v in L], lv[n]


def _box(n, S, h, rng):
    """two parallel Pratt planes at y=0 and y=W, transverse + plan-diagonal bracing; vertical = z."""
    b = B(); s = S / n; W = h * rng.uniform(.6, 1.2)
    bot = [[b.node(i * s, y, 0) for i in range(n + 1)] for y in (0, W)]
    top = [[b.node(i * s, y, h) for i in range(n + 1)] for y in (0, W)]
    for p in range(2):
        for i in range(n):
            b.bar(bot[p][i], bot[p][i + 1], "chord"); b.bar(top[p][i], top[p][i + 1], "chord")
            b.bar(*((top[p][i], bot[p][i + 1]) if i < n / 2 else (bot[p][i], top[p][i + 1])), "web")
        for i in range(n + 1):
            b.bar(bot[p][i], top[p][i], "web")
    for i in range(n + 1):
        b.bar(bot[0][i], bot[1][i], "web"); b.bar(top[0][i], top[1][i], "web")
    for i in range(n):
        b.bar(bot[0][i], bot[1][i + 1], "web"); b.bar(top[0][i], top[1][i + 1], "web")
    b.bar(bot[0][0], top[1][0], "web"); b.bar(bot[0][n], top[1][n], "web")
    return b, [bot[0][0], bot[1][0], bot[0][n], bot[1][n]], top[0] + top[1]


def make_aug(fam, i, flip_p):
    """§4k round 5: extra training sample of a vertical-post panel family; flip_p=0.5 (exp) or 0 (control)."""
    import hashlib
    _FLIP["p"] = flip_p
    _FLIP["rng"] = np.random.default_rng(int(hashlib.md5(f"flip-{fam}-{i}".encode()).hexdigest()[:16], 16))
    try:
        r = make(fam, "small", 10 ** 7 + i)
    finally:
        _FLIP["p"] = 0.0
    r["aug"] = "flip" if flip_p else "plain"
    return r


def make(fam, size, i, lam=None):
    lam = (LAMBDA if lam is None else lam) * LAM_FAM.get(fam, 1.0)
    import hashlib
    rng = np.random.default_rng(int(hashlib.md5(f"truss-{fam}-{size}-{i}".encode()).hexdigest()[:16], 16))
    lo, hi = PANELS[size]
    for _ in range(50):
        n = int(rng.integers(lo, hi + 1))
        S = float(rng.uniform(8, 40) * (n / 7) ** .5)
        sd = float(rng.uniform(4, 14)); h = S / sd
        q = float(np.exp(rng.uniform(np.log(5e3), np.log(40e3)))) * lam  # N/m line load (2D) or N/m^2 * s (3D)
        if fam == "space":
            b, sup, loaded, trib = _space(n, S, h * .6, rng)
            fixed = [[False] * 3 for _ in b.X]
            for v in sup:
                fixed[v] = PIN
            P = q / 4 * trib
        elif fam == "tower":
            b, sup, loaded, topl = _tower(n, S, h, rng)
            fixed = [[False] * 3 for _ in b.X]
            for v in sup:
                fixed[v] = PIN
            P = q * S / max(1, len(loaded))
        elif fam == "box":
            b, sup, loaded = _box(n, S, h, rng)
            fixed = [[False] * 3 for _ in b.X]
            fixed[sup[0]] = PIN; fixed[sup[1]] = [True, False, True]; fixed[sup[2]] = [False, True, True]
            fixed[sup[3]] = [False, False, True]
            P = q * S / max(1, len(loaded))
        else:
            b, bot, top = {"fink": _fink, "random": _random}.get(fam, lambda *a: _flat(fam, *a))(n, S, h, rng)
            fixed = [list(PLANE) for _ in b.X]
            fixed[bot[0]] = PIN; fixed[bot[-1] if fam != "random" else bot[1]] = ROLL
            loaded = top if rng.random() < .7 else bot
            P = q * S / max(1, len(loaded))
        X = np.array(b.X); M = np.array(b.M)
        Ac = float(np.exp(rng.uniform(np.log(1e-3), np.log(8e-3)))); Aw = Ac * rng.uniform(.3, 1)
        A = np.array([Ac if g == "chord" else Aw for g in b.grp]) * rng.uniform(.8, 1.25, len(M))
        # circular hollow tubes: per member group pick the outer diameter so typical slenderness L/r ~ U[60,160]
        L = np.linalg.norm(X[M[:, 1]] - X[M[:, 0]], axis=1)
        r = np.empty(len(M))
        for g in ("chord", "web"):
            k = np.array([x == g for x in b.grp])
            if k.any():
                d = np.median(L[k]) / (.35 * rng.uniform(60, 160))
                r[k] = np.where(A[k] < np.pi * d ** 2 / 4, .35 * d, np.sqrt(A[k] / (4 * np.pi)))  # too thick -> solid
        I = A * r ** 2
        F = np.zeros_like(X)
        lat = rng.uniform(0, .1)
        for v in loaded:
            if fixed[v][1] and fam not in THREE_D:
                continue
            w = P * rng.uniform(.5, 1.5)
            if fam == "tower":  # wind-dominated: lateral share U[0.2,1] of the gravity-scale load
                F[v] = [w * lat * 5 + w * .2, 0, -w * .3]
            elif fam in THREE_D:
                F[v] = [lat * w, 0, -w]
            else:
                F[v] = [lat * w, -w, 0]
        try:
            j = judge(X, M, A, np.array(fixed), F, S, I)
        except np.linalg.LinAlgError:
            continue
        return {"fam": fam, "size": size, "i": i, "n": n, "span": S, "depth": float(np.ptp(X[:, 2 if fam in THREE_D else 1])), "grp": b.grp,
                "X": X.round(5).tolist(), "M": M.tolist(), "A": A.tolist(), "I": I.tolist(), "fixed": np.array(fixed).astype(int).tolist(),
                "F": F.round(3).tolist(), "span_depth": S / max(1e-9, float(np.ptp(X[:, 0] if fam == "tower" else X[:, 2 if fam in THREE_D else 1]))),
                "max_slender": float((L / r).max()), **j}
    raise RuntimeError(f"no stable truss for {fam} {size} {i}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000); ap.add_argument("--size", default="small")
    ap.add_argument("--out"); ap.add_argument("--calibrate", action="store_true"); ap.add_argument("--lam", type=float)
    ap.add_argument("--fams", nargs="*")
    a = ap.parse_args()
    if a.calibrate:
        res = {}
        for lam in (0.3, 0.4, 0.5, 0.6):
            u = {f: np.mean([make(f, "small", 10 ** 6 + i, lam)["unsafe"] for i in range(300)]) for f in FAMILIES}
            res[lam] = {"pooled": float(np.mean(list(u.values()))), **{f: float(v) for f, v in u.items()}}
            print(lam, json.dumps(res[lam]), flush=True)
        Path("results/truss").mkdir(parents=True, exist_ok=True)
        Path("results/truss/calibration.json").write_text(json.dumps(res, indent=1))
        return
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); t = time.time()
    with out.open("w") as f:
        for fam in (a.fams or FAMILIES):
            for i in range(a.n):
                f.write(json.dumps(make(fam, a.size, i, a.lam)) + "\n")
            print(fam, f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
