"""Whole-structure physics checker wrapping BrickGPT's BrickStructure.

check(text) -> CheckResult(ok, failure, first_bad_index, message)
failure in {None, 'format', 'collision', 'out_of_bounds', 'floating', 'unstable'}
Mechanics via Gurobi when a license works; otherwise falls back to connectivity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import numpy as np

from . import _VENDOR  # noqa: F401  (ensures vendor path)
from brickgpt.data import Brick, BrickStructure
from brickgpt.data.brick_library import dimensions_to_brick_id

_LINE = re.compile(r"(\d+)x(\d+) \((\d+),(\d+),(\d+)\)")


@dataclass
class CheckResult:
    ok: bool
    failure: str | None
    first_bad_index: int | None  # 0-based brick index
    message: str
    n_bricks: int = 0
    method: str = ""  # 'gurobi' | 'connectivity' | ''

    def to_dict(self):
        return asdict(self)


class GurobiUnavailable(RuntimeError):
    pass


_gurobi_ok: bool | None = None
_gurobi_env = None


def _quiet_env():
    """One shared Gurobi Env with all logging off (license banners never reach logs)."""
    global _gurobi_env
    if _gurobi_env is None:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()
        _gurobi_env = env
    return _gurobi_env


def gurobi_available() -> bool:
    """True if gurobipy imports and a license lets us build/solve a tiny model."""
    global _gurobi_ok
    if _gurobi_ok is None:
        try:
            import gurobipy as gp
            # >2000 vars/constraints: the pip-bundled size-limited license fails here, so this
            # only passes with a real (e.g. WLS academic) license.
            with gp.Model(env=_quiet_env()) as m:
                x = m.addVars(2500)
                m.addConstrs(x[i] >= 0 for i in range(2500))
                m.optimize()
            _gurobi_ok = True
        except Exception:
            _gurobi_ok = False
    return _gurobi_ok


def extract_bricks_text(text: str) -> str:
    """Keep only brick lines (models may wrap output in prose/code fences)."""
    return "\n".join(m.group(0) for m in _LINE.finditer(text))


def parse(text: str, world_dim: int = 20, strict: bool = False):
    """Return (bricks, error_message, bad_index)."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not strict:
        lines = [l for l in lines if not l.startswith("```")]
    bricks = []
    for i, line in enumerate(lines):
        if strict:
            m = _LINE.fullmatch(line)
            if m is None:
                return None, f"Line {i + 1} '{line[:40]}' is not in the format 'HxW (x,y,z)'.", i
            ms = [m]
        else:  # every brick on the line counts (models sometimes emit one space-separated line)
            ms = list(_LINE.finditer(line))
        for m in ms:
            h, w = int(m.group(1)), int(m.group(2))
            try:
                dimensions_to_brick_id(h, w)
            except ValueError:
                return None, f"Brick {len(bricks) + 1} '{m.group(0)}' has unsupported dimensions {h}x{w}.", len(bricks)
            bricks.append(Brick.from_txt(m.group(0)))
    n_found = len(_LINE.findall(text))
    if bricks and len(bricks) != n_found:  # invariant: never silently drop bricks
        return None, f"Parsed {len(bricks)} bricks but found {n_found} brick patterns.", None
    if not bricks:
        return None, "No bricks found. Output one brick per line as 'HxW (x,y,z)'.", None
    return bricks, None, None


def gurobi_scores(s: BrickStructure, tol: float = 1e-6, time_limit: float | None = 120,
                  threads: int | None = None, lp: bool = False, grb_params: dict | None = None) -> np.ndarray:
    """Per-voxel stability scores (>=1 means unstable) via the patched BrickGPT force analysis."""
    from brickgpt.data.brick_library import brick_library
    from .stability_patched import stability_score, StabilityConfig
    if grb_params is None:  # NumericFocus=3: 4.2x faster, identical verdicts on 80 test structures (DEBUG_LOG)
        grb_params = {"NumericFocus": 3}
    cfg = StabilityConfig(world_dimension=(s.world_dim,) * 3, tol=tol, time_limit=time_limit, threads=threads,
                          env=_quiet_env(), lp=lp, grb_params=grb_params)
    scores, *_ = stability_score(s.to_json(), brick_library, cfg)
    return scores


def check(text: str, world_dim: int = 20, use_gurobi: bool | str = "auto", strict: bool = False,
          gurobi_kwargs: dict | None = None) -> CheckResult:
    """use_gurobi:
      True    -> Gurobi mechanics (main results). Raises GurobiUnavailable if no working license;
                 a per-structure solver error/timeout returns failure='checker_error' (never a silent fallback).
      False   -> connectivity only (appendix ablation / tests).
      'auto'  -> Gurobi if licensed else connectivity (dev convenience only; .method records which)."""
    bricks, err, bad = parse(text, world_dim, strict)
    if bricks is None:
        return CheckResult(False, "format", bad, err)
    n = len(bricks)

    # Bounds (must come before building the voxel grid to avoid index errors)
    for i, b in enumerate(bricks):
        if not (b.x >= 0 and b.y >= 0 and b.x + b.h <= world_dim and b.y + b.w <= world_dim and 0 <= b.z < world_dim):
            return CheckResult(False, "out_of_bounds", i,
                               f"Brick {i + 1} '{b!r}' is outside the {world_dim}x{world_dim}x{world_dim} grid.", n)

    s = BrickStructure([], world_dim=world_dim)
    for i, b in enumerate(bricks):
        if s.brick_collides(b):
            return CheckResult(False, "collision", i,
                               f"Brick {i + 1} '{b!r}' overlaps a previously placed brick.", n)
        s.add_brick(b)

    for i, b in enumerate(s.bricks):
        if s.brick_floats(b):
            return CheckResult(False, "floating", i,
                               f"Brick {i + 1} '{b!r}' is floating: nothing above or below it.", n)

    if use_gurobi is True and not gurobi_available():
        raise GurobiUnavailable("use_gurobi=True but no working Gurobi license (set GRB_LICENSE_FILE / WLS secrets)")
    method = "gurobi" if (use_gurobi is True or (use_gurobi == "auto" and gurobi_available())) else "connectivity"
    if method == "gurobi":
        try:
            scores = gurobi_scores(s, **(gurobi_kwargs or {}))
        except Exception as e:
            return CheckResult(False, "checker_error", None, f"Gurobi failed: {type(e).__name__}: {str(e)[:120]}",
                               n, "gurobi")
    else:
        scores = s.connectivity_scores()
    if np.max(scores) >= 1:
        idx = next(i for i, b in enumerate(s.bricks) if np.any(scores[b.slice] >= 1))
        what = "would collapse under gravity" if method == "gurobi" else "is not connected to the ground"
        return CheckResult(False, "unstable", idx,
                           f"Structure is unstable: brick {idx + 1} '{s.bricks[idx]!r}' {what}.", n, method)
    return CheckResult(True, None, None, "OK: structure is valid and stable.", n, method)
