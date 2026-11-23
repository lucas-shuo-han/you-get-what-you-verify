"""Brick tokenizer: one brick = 4 tokens (type, X, Y, Z). < 100 tokens total.

Sequence grammar (all arms):
  BOS CAT_c  { brick | error-event }*  EOS
  error-event (R+F, R+F~):  bad_brick FB  FAIL_t  D d d d  BACK
  error-event (R):          bad_brick BACK
BACK means "retract the previous brick". Feedback = failure type + 1-based index of the bad brick (3 digits).
"""
from __future__ import annotations

from .. import _VENDOR  # noqa: F401  (adds vendor BrickGPT to sys.path)
from brickgpt.data.brick_structure import Brick

BRICK_TYPES = ["2x4", "4x2", "2x6", "6x2", "1x2", "2x1", "1x4", "4x1", "1x6", "6x1", "1x8", "8x1", "1x1", "2x2"]
FAIL_TYPES = ["out_of_bounds", "collision", "floating", "unstable"]
N_CAT = 21
GRID = 20

SPECIAL = ["PAD", "BOS", "EOS", "FB", "BACK"]
TOKENS = (SPECIAL + [f"CAT_{i}" for i in range(N_CAT)] + [f"T_{t}" for t in BRICK_TYPES]
          + [f"X{i}" for i in range(GRID)] + [f"Y{i}" for i in range(GRID)] + [f"Z{i}" for i in range(GRID)]
          + [f"FAIL_{f}" for f in FAIL_TYPES] + [f"D{i}" for i in range(10)])
ID = {t: i for i, t in enumerate(TOKENS)}
PAD, BOS, EOS, FB, BACK = (ID[t] for t in SPECIAL)
VOCAB_SIZE = len(TOKENS)


def brick_tokens(b: Brick) -> list[int]:
    return [ID[f"T_{b.h}x{b.w}"], ID[f"X{b.x}"], ID[f"Y{b.y}"], ID[f"Z{b.z}"]]


def feedback_tokens(fail: str, index1: int) -> list[int]:
    s = f"{min(index1, 999):03d}"
    return [FB, ID[f"FAIL_{fail}"]] + [ID[f"D{c}"] for c in s]


def decode(ids: list[int]) -> tuple[list[Brick], dict]:
    """Token ids -> resolved brick list (BACK retracts, feedback skipped) + stats."""
    bricks, i, n_back, n_bad_tok = [], 0, 0, 0
    names = [TOKENS[t] if 0 <= t < VOCAB_SIZE else "?" for t in ids]
    while i < len(names):
        t = names[i]
        if t in ("EOS", "PAD"):
            break
        if t == "BOS" or t.startswith("CAT_"):
            i += 1
        elif t == "BACK":
            if bricks:
                bricks.pop()
            n_back += 1
            i += 1
        elif t == "FB":
            i += 5  # FB FAIL_x D D D
        elif t.startswith("T_") and i + 3 < len(names) and names[i + 1][0] == "X" and names[i + 2][0] == "Y" \
                and names[i + 3][0] == "Z":
            h, w = map(int, t[2:].split("x"))
            bricks.append(Brick(h=h, w=w, x=int(names[i + 1][1:]), y=int(names[i + 2][1:]), z=int(names[i + 3][1:])))
            i += 4
        else:
            n_bad_tok += 1
            i += 1
    return bricks, {"n_back": n_back, "n_bad_tokens": n_bad_tok}


def bricks_to_text(bricks: list[Brick]) -> str:
    return "\n".join(f"{b.h}x{b.w} ({b.x},{b.y},{b.z})" for b in bricks)
