"""§4j linear-elastic pin-jointed truss, direct stiffness method (3D; 2D trusses fix the z DOF of every node).

Material fixed: steel E = 200 GPa, allowable stress 165 MPa (= 250/1.5). Sections: circular hollow tubes, I given per member (default solid circle A^2/(4 pi) if omitted).
Unsafe iff any of:  |sigma| > SIGMA_ALLOW;  compression N > pi^2 E I / L^2 (pinned, K=1);  max |u| > span / 250.
"""
from __future__ import annotations

import numpy as np

E = 200e9
SIGMA_ALLOW = 165e6


def inertia(A):
    return np.asarray(A) ** 2 / (4 * np.pi)


def solve(X, members, A, fixed, F):
    """X [n,3] m; members [m,2] int; A [m] m^2; fixed [n,3] bool; F [n,3] N.
    -> (u [n,3] m, N [m] axial force, + = tension). Raises np.linalg.LinAlgError for a mechanism."""
    X = np.asarray(X, float); members = np.asarray(members, int); A = np.asarray(A, float)
    n = len(X)
    d = X[members[:, 1]] - X[members[:, 0]]
    L = np.linalg.norm(d, axis=1)
    c = d / L[:, None]
    k = E * A / L
    K = np.zeros((3 * n, 3 * n))
    for e, (i, j) in enumerate(members):
        kk = k[e] * np.outer(c[e], c[e])
        for a, b, s in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            K[3 * a:3 * a + 3, 3 * b:3 * b + 3] += s * kk
    free = ~np.asarray(fixed, bool).reshape(-1)
    Kf = K[np.ix_(free, free)]
    if np.linalg.cond(Kf) > 1e12:
        raise np.linalg.LinAlgError("mechanism / ill-conditioned stiffness")
    u = np.zeros(3 * n)
    u[free] = np.linalg.solve(Kf, np.asarray(F, float).reshape(-1)[free])
    u = u.reshape(n, 3)
    N = k * ((u[members[:, 1]] - u[members[:, 0]]) * c).sum(1)
    return u, N


def judge(X, members, A, fixed, F, span, I=None):
    """-> dict with unsafe flag, the three utilisations (>1 = violated) and which modes fail."""
    u, N = solve(X, members, A, fixed, F)
    L = np.linalg.norm(X[members[:, 1]] - X[members[:, 0]], axis=1)
    stress = np.abs(N) / A / SIGMA_ALLOW
    pcr = np.pi ** 2 * E * (inertia(A) if I is None else np.asarray(I)) / L ** 2
    buck = np.where(N < 0, -N / pcr, 0.0)
    disp = np.abs(u).max() / (span / 250)
    util = {"stress": float(stress.max()), "buckling": float(buck.max()), "disp": float(disp)}
    return {"unsafe": int(max(util.values()) > 1), "util": util, "modes": [k for k, v in util.items() if v > 1],
            "comp": (N < 0).astype(int).tolist()}
