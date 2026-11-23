"""§4g screener tests: conformal coverage, graph construction, label determinism, verdict rules."""
import numpy as np
import pytest


def test_conformal_one_sided_coverage_synthetic():
    from brickexp.screen.conformal import threshold, metrics
    rng = np.random.default_rng(0)
    for alpha in (0.01, 0.05):
        misses = []
        for _ in range(400):
            n1 = 300
            cal = rng.beta(5, 2, n1)          # unstable scores (exchangeable with test unstables)
            tau = threshold(cal, np.ones(n1, bool), alpha)
            test = rng.beta(5, 2, 2000)
            misses.append(metrics(test, np.ones(2000, bool), tau)["miss"])
        assert np.mean(misses) <= alpha + 0.002, (alpha, np.mean(misses))
        assert np.mean(misses) >= alpha * 0.5  # not vacuous


def test_conformal_infeasible_small_n_saves_nothing():
    from brickexp.screen.conformal import threshold, metrics
    tau = threshold([0.9, 0.8, 0.1], [1, 1, 0], 0.05)  # n1=2 < 1/alpha-1
    assert tau == -np.inf
    m = metrics([0.0, 0.5], [0, 1], tau)
    assert m["saved"] == 0 and m["miss"] == 0


def test_threshold_uses_only_unstable_and_kth_smallest():
    from brickexp.screen.conformal import threshold
    s = np.r_[np.linspace(0.1, 1, 199), 0.0]  # stable sample with score 0 must be ignored
    y = np.r_[np.ones(199), 0]
    assert threshold(s, y, 0.01) == pytest.approx(np.sort(s[:199])[1])  # k = floor(.01*200)=2 -> 2nd smallest
    s2 = np.sort(s[:199]); assert threshold(s, y, 0.05) == pytest.approx(s2[9])


def test_brick_graph_edges():
    from brickgpt.data.brick_structure import Brick
    from brickexp.screen.gnn import brick_graph, collate, Screener
    b = [Brick(h=2, w=4, x=0, y=0, z=0), Brick(h=2, w=4, x=2, y=0, z=0),  # side by side (lateral, side 4)
         Brick(h=2, w=2, x=1, y=1, z=1),                                    # on both (area 1x2 each)
         Brick(h=1, w=1, x=10, y=10, z=1)]                                   # candidate, floating, no edges
    f, e, w = brick_graph(b)
    E = {(s, d, r): x for (s, d, r), x in zip(e.tolist(), w.tolist())}
    assert E[(0, 1, 2)] == E[(1, 0, 2)] == pytest.approx(4 / 16)
    assert E[(0, 2, 0)] == pytest.approx(2 / 16) and E[(2, 0, 1)] == pytest.approx(2 / 16)
    assert E[(1, 2, 0)] == pytest.approx(2 / 16)
    assert not any(3 in (s, d) for s, d, _ in E) and len(E) == 6
    assert f[3, 6] == 1 and f[:3, 6].sum() == 0 and f[0, 5] == 1 and f[2, 5] == 0
    out = Screener(16, 2)(*collate([(f, e, w), brick_graph(b[:2])]))
    assert out.shape == (2,) and torch_finite(out)


def torch_finite(t):
    import torch
    return bool(torch.isfinite(t).all())


def test_label_samples_deterministic_and_pass_cheap_checks():
    from brickexp.screen import labels as Lb
    from brickexp.tiny.gen_data import cheap_fail
    from brickgpt.data.brick_structure import BrickStructure
    try:
        recs = Lb.pool("cat")
    except Exception:
        pytest.skip("dataset not available")
    assert {c for _, c, _ in recs} == set(Lb.HELD)
    for i in range(5):
        m1, b1 = Lb.make_sample("cat", i, recs)
        m2, b2 = Lb.make_sample("cat", i, recs)
        assert m1 == m2 and [(x.h, x.w, x.x, x.y, x.z) for x in b1] == [(x.h, x.w, x.x, x.y, x.z) for x in b2]
        assert len(b1) <= Lb.SIZES["cat"][1]
        assert cheap_fail(BrickStructure(b1[:-1], world_dim=20), b1[-1]) is None
    assert not ({c for _, c, _ in Lb.pool("id")} & set(Lb.HELD))


def test_verdict_rules():
    from brickexp.screen.run import verdict
    idm = {"miss": 0.01, "saved": 0.6}
    assert verdict(idm, {"miss": 0.03, "saved": 0.6}, {1000: {"miss": 0.009, "saved": 0.4}}, 0.01)["verdict"] == "REAL_fixable"
    assert verdict(idm, {"miss": 0.01, "saved": 0.2}, {1000: {"miss": 0.02, "saved": 0.4}}, 0.01)["verdict"] == "REAL_unfixable"
    assert verdict(idm, {"miss": 0.008, "saved": 0.55}, {}, 0.01)["verdict"] == "NOT_REAL"
    assert verdict(idm, {"miss": 0.015, "saved": 0.55}, {}, 0.01)["verdict"] == "INCONCLUSIVE"
