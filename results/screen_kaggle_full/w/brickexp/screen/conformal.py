"""§4g one-sided split-conformal screening.

Score s = P(unstable) from the screener. Calibrate on the UNSTABLE calibration samples only (n1 of them):
    k = floor(alpha * (n1 + 1)); tau = k-th smallest unstable calibration score (tau = -inf if k < 1)
    judged safe  <=>  s < tau   (judged safe = solver call skipped)
For a new unstable sample exchangeable with the calibration unstables, P(s_new < tau) <= k/(n1+1) <= alpha,
i.e. miss rate := P(judged safe | unstable) <= alpha (marginal, finite-sample). If n1 < 1/alpha - 1 no tau is
valid and nothing is judged safe (0 savings) -- the honest small-N outcome.
"""
from __future__ import annotations

import numpy as np


def threshold(cal_scores, cal_unstable, alpha):
    s1 = np.sort(np.asarray(cal_scores)[np.asarray(cal_unstable, bool)])
    k = int(np.floor(alpha * (len(s1) + 1)))
    return -np.inf if k < 1 else float(s1[k - 1])


def metrics(scores, unstable, tau):
    scores, unstable = np.asarray(scores), np.asarray(unstable, bool)
    safe = scores < tau  # direction fixed: s = P(unstable); low score = safe
    return {"auc": auc(scores, unstable), "infeasible": bool(tau == -np.inf),
            "precision_safe": float(1 - unstable[safe].mean()) if safe.any() else float("nan"),"miss": float(safe[unstable].mean()) if unstable.any() else float("nan"),
            "saved": float(safe.mean()),
            "unsafe_among_safe": float(unstable[safe].mean()) if safe.any() else 0.0,
            "n": int(len(scores)), "n_unstable": int(unstable.sum()), "tau": tau}


def auc(scores, y):
    """AUC of P(unstable) for the unstable class (Mann-Whitney, ties 1/2); separates ranking from threshold failure."""
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = _rank_avg(np.asarray(scores, float)); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def id_tolerance(alpha, n_cal1, n_test1, z=3.0):
    """Sampling tolerance for the ID sanity check (miss <= alpha + tol): calibration + test binomial noise."""
    return float(z * np.sqrt(alpha * (1 - alpha) * (1 / max(n_cal1, 1) + 1 / max(n_test1, 1))))


def _rank_avg(a):
    """1-based ranks with ties averaged (numpy-only equivalent of scipy.stats.rankdata)."""
    o = np.argsort(a, kind="mergesort"); s = a[o]
    first = np.r_[True, s[1:] != s[:-1]]
    grp = np.cumsum(first) - 1
    starts = np.flatnonzero(first); ends = np.r_[starts[1:], len(a)]
    avg = (starts + ends + 1) / 2.0
    r = np.empty(len(a)); r[o] = avg[grp]
    return r
