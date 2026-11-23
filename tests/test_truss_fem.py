"""§4j solver validation against closed-form trusses."""
import numpy as np

from brickexp.truss.fem import E, solve

PIN2D = [True, True, True]


def test_two_bar_symmetric():
    # apex at (0,h), supports at (+-b,0); vertical load P at apex
    b, h, P, A = 3.0, 4.0, 1e5, 1e-3
    X = [[-b, 0, 0], [b, 0, 0], [0, h, 0]]
    m = [[0, 2], [1, 2]]
    fixed = [PIN2D, PIN2D, [False, False, True]]
    F = [[0, 0, 0], [0, 0, 0], [0, -P, 0]]
    u, N = solve(X, m, [A, A], fixed, F)
    L = np.hypot(b, h); s = h / L
    assert np.allclose(N, -P / (2 * s), rtol=1e-9)                  # both bars in compression
    assert np.isclose(u[2, 1], -P * L / (2 * E * A * s ** 2), rtol=1e-9)


def test_triangle_pin_roller():
    # simple triangular truss: span 2a, height h, load P at apex; bottom chord tension P/(2 tan t)
    a, h, P, A = 4.0, 3.0, 2e5, 2e-3
    X = [[0, 0, 0], [2 * a, 0, 0], [a, h, 0]]
    m = [[0, 1], [0, 2], [1, 2]]
    fixed = [PIN2D, [False, True, True], [False, False, True]]
    F = [[0, 0, 0], [0, 0, 0], [0, -P, 0]]
    u, N = solve(X, m, [A] * 3, fixed, F)
    t = np.arctan2(h, a)
    assert np.isclose(N[0], P / (2 * np.tan(t)), rtol=1e-9)
    assert np.allclose(N[1:], -P / (2 * np.sin(t)), rtol=1e-9)


def test_tripod_3d():
    # three legs from apex (0,0,H) to base circle radius R; vertical load P; leg force P/(3 cos phi)
    R, H, P, A = 2.0, 5.0, 3e5, 1e-3
    base = [[R * np.cos(t), R * np.sin(t), 0] for t in (0, 2 * np.pi / 3, 4 * np.pi / 3)]
    X = base + [[0, 0, H]]
    m = [[0, 3], [1, 3], [2, 3]]
    fixed = [[True] * 3] * 3 + [[False] * 3]
    F = [[0, 0, 0]] * 3 + [[0, 0, -P]]
    u, N = solve(X, m, [A] * 3, fixed, F)
    L = np.hypot(R, H)
    assert np.allclose(N, -P * L / (3 * H), rtol=1e-9)
    assert np.isclose(u[3, 2], -P * L ** 3 / (3 * E * A * H ** 2), rtol=1e-9)


def test_mechanism_rejected():
    import pytest
    X = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    m = [[0, 1], [1, 2], [2, 3], [3, 0]]  # square without diagonal = mechanism
    fixed = [PIN2D, [False, True, True], [False, False, True], [False, False, True]]
    with pytest.raises(np.linalg.LinAlgError):
        solve(X, m, [1e-3] * 4, fixed, [[0, 0, 0], [0, 0, 0], [1e3, 0, 0], [0, 0, 0]])
