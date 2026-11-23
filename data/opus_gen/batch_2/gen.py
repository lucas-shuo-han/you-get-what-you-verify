"""Batch 2 of flagship-model distillation data: 50 prompts from data/opus_gen/prompts_2.jsonl.
Shapes designed from caption only (reference bricks never read). Reuses tools/opus_pilot.py tiling + rendering.
python data/opus_gen/batch_2/gen.py [idx ...]
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from brickexp.checker import check  # noqa: E402
from distill_filter import feats, keep  # noqa: E402
import opus_pilot as P  # noqa: E402

PREF = P.PREF


def box(V, x0, x1, y0, y1, z0, z1, hollow=False):
    P.box(V, x0, x1, y0, y1, z0, z1, hollow)


def rm(V, x0, x1, y0, y1, z0, z1):
    for x in range(x0, x1):
        for y in range(y0, y1):
            for z in range(z0, z1):
                V.discard((x, y, z))


def disc(r, cx=10, cy=10):
    return {(x, y) for x in range(20) for y in range(20) if (x + .5 - cx) ** 2 + (y + .5 - cy) ** 2 <= r * r}


def put(V, cells, z0, z1):
    for x, y in cells:
        for z in range(z0, z1):
            V.add((x, y, z))


def ring(r, t=2, cx=10, cy=10):
    return disc(r, cx, cy) - disc(r - t, cx, cy)


def hexa(r, cx=10, cy=10):
    out = set()
    for x in range(20):
        for y in range(20):
            dx, dy = abs(x + .5 - cx), abs(y + .5 - cy)
            if dy <= r * .866 and dx + dy * .577 <= r:
                out.add((x, y))
    return out


# ---------------------------------------------------------------- generic furniture
def table(V, X, Y, H, leg=2, top=2, apron=True, x0=None, y0=None, inset=0):
    x0 = (20 - X) // 2 if x0 is None else x0; y0 = (20 - Y) // 2 if y0 is None else y0
    for lx in (x0 + inset, x0 + X - leg - inset):
        for ly in (y0 + inset, y0 + Y - leg - inset):
            box(V, lx, lx + leg, ly, ly + leg, 0, H - top)
    if apron:
        box(V, x0 + inset, x0 + X - inset, y0 + inset, y0 + Y - inset, H - top - 1, H - top, hollow=True)
    box(V, x0, x0 + X, y0, y0 + Y, H - top, H)


def chair(V, S=8, D=8, seat=5, back=7, leg=2, x0=6, y0=6, back_style="solid", stretch=True):
    for lx in (x0, x0 + S - leg):
        for ly in (y0, y0 + D - leg):
            box(V, lx, lx + leg, ly, ly + leg, 0, seat - 1)
    if stretch:
        box(V, x0, x0 + S, y0, y0 + D, seat - 2, seat - 1, hollow=True)
    box(V, x0, x0 + S, y0, y0 + D, seat - 1, seat + 1)          # 2-thick seat
    zb = seat + 1
    yb = y0 + D - 2
    if back_style == "solid":
        box(V, x0, x0 + S, yb, yb + 2, zb, zb + back)
    elif back_style == "slats":         # posts + horizontal rails
        box(V, x0, x0 + 2, yb, yb + 2, zb, zb + back); box(V, x0 + S - 2, x0 + S, yb, yb + 2, zb, zb + back)
        for z in range(zb + 1, zb + back, 2):
            box(V, x0, x0 + S, yb, yb + 2, z, z + 1)
        box(V, x0, x0 + S, yb, yb + 2, zb + back - 1, zb + back)
    elif back_style == "vslats":        # vertical slats between rails
        for x in range(x0, x0 + S, 2):
            box(V, x, x + 1, yb, yb + 2, zb, zb + back - 1)
        box(V, x0 + S - 1, x0 + S, yb, yb + 2, zb, zb + back - 1)
        box(V, x0, x0 + S, yb, yb + 2, zb + back - 1, zb + back)
    elif back_style == "step":          # stepped: each pair of layers narrower
        for k, z in enumerate(range(zb, zb + back)):
            s = min(k // 2, S // 2 - 2)
            box(V, x0 + s, x0 + S - s, yb, yb + 2, z, z + 1)


def sofa(V, L=16, D=8, seat=3, back=6, arm=2, arm_h=5, x0=2, y0=6, legs=True):
    z = 0
    if legs:
        for lx in (x0, x0 + L - 2):
            for ly in (y0, y0 + D - 2):
                box(V, lx, lx + 2, ly, ly + 2, 0, 1)
        z = 1
    box(V, x0, x0 + L, y0, y0 + D, z, seat, hollow=True)
    box(V, x0, x0 + L, y0, y0 + D, seat, seat + 1)                 # cushion deck
    for k in range(0, L - 2 * arm, 4):                              # cushion seams = separate cushions
        box(V, x0 + arm + k, x0 + arm + min(k + 4, L - 2 * arm) - (1 if k + 4 < L - 2 * arm else 0), y0, y0 + D - 2, seat + 1, seat + 2)
    box(V, x0, x0 + L, y0 + D - 2, y0 + D, seat + 1, seat + 1 + back)  # back
    if arm:
        for ax in (x0, x0 + L - arm):
            box(V, ax, ax + arm, y0, y0 + D - 2, seat + 1, arm_h + 1)


# ---------------------------------------------------------------- vehicles
def car(V, L=16, W=8, y0=2, x0=6, body=2, cab=(4, 11), cab_h=2, tiers=1, hood_step=True):
    for yy in (y0 + 1, y0 + L - 4):
        box(V, x0 - 1 if W < 10 else x0, x0 + W + (1 if W < 10 else 0), yy, yy + 3, 0, 1)   # axles+wheels
    box(V, x0, x0 + W, y0, y0 + L, 1, 1 + body)
    zc = 1 + body
    a, b = cab
    box(V, x0, x0 + W, y0 + a, y0 + b, zc, zc + cab_h, hollow=True)
    for y in range(y0 + a + 1, y0 + b - 1):         # side windows
        for z in range(zc + 1, zc + cab_h):
            V.discard((x0, y, z)); V.discard((x0 + W - 1, y, z))
    box(V, x0, x0 + W, y0 + a, y0 + b, zc + cab_h, zc + cab_h + 1)
    for t in range(1, tiers):
        box(V, x0 + t, x0 + W - t, y0 + a + t, y0 + b - t, zc + cab_h + t, zc + cab_h + t + 1)
    if hood_step:
        box(V, x0 + 1, x0 + W - 1, y0 + b, y0 + L - 1, zc, zc + 1)     # raised hood
        box(V, x0 + 1, x0 + W - 1, y0 + 1, y0 + a, zc, zc + 1)         # trunk lid


def bus(V, L=20, W=7, H=6, raised=None, x0=6, windows=True):
    box(V, x0 - 1, x0 + W + 1, 2, 4, 0, 1); box(V, x0 - 1, x0 + W + 1, L - 5, L - 3, 0, 1)
    box(V, x0, x0 + W, 0, L, 1, 2)
    box(V, x0, x0 + W, 0, L, 2, H, hollow=True)
    if windows:
        for y in range(2, L - 2):
            if y % 3 != 1:
                V.discard((x0, y, H - 2)); V.discard((x0 + W - 1, y, H - 2))
        for x in range(x0 + 2, x0 + W - 2):
            V.discard((x, L - 1, H - 2))            # windshield
    box(V, x0, x0 + W, 0, L, H, H + 1)

    if raised:
        a, b = raised
        box(V, x0 + 1, x0 + W - 1, a, b, H + 1, H + 2)


def train(V, L=18, W=6, H=4, x0=7, tiers=3, cab=None, e=1):
    box(V, x0, x0 + W, 0, L, 0, 1)
    for y in range(1, L - 2, 5):
        box(V, x0 - 1, x0 + W + 1, y, y + 2, 0, 1)
    box(V, x0, x0 + W, 0, L, 1, 1 + H, hollow=True)
    for y in range(2, L - 2):
        if y % 4 in (1, 2):
            V.discard((x0, y, H - 1)); V.discard((x0 + W - 1, y, H - 1))
    box(V, x0 - e, x0 + W + e, 0, L, 1 + H, 2 + H)
    for t in range(1, tiers):
        box(V, x0 - 1 + t, x0 + W + 1 - t, t, L - t, 1 + H + t, 2 + H + t)


def vessel(V, L=16, B=6, x0=7, y0=2, deck_z=3, tiers=((3, 12, 2),), mast=None, step_sides=True):
    for z in range(deck_z):
        s = (deck_z - 1 - z) if step_sides else 0
        box(V, x0 + min(s, B // 2 - 1), x0 + B - min(s, B // 2 - 1), y0 + s, y0 + L - 2 * s if s else y0 + L, z, z + 1)
    box(V, x0, x0 + B, y0, y0 + L, deck_z, deck_z + 1)
    z = deck_z + 1
    for k, (a, b, h) in enumerate(tiers):
        i = k + 1
        box(V, x0 + i, x0 + B - i, y0 + a, y0 + b, z, z + h, hollow=(h > 1 and B - 2 * i > 2))
        box(V, x0 + i, x0 + B - i, y0 + a, y0 + b, z + h - 1, z + h)
        z += h
    if mast:
        mx, my, mh = mast
        box(V, mx, mx + 2, my, my + 2, z, z + mh)


def revolve(V, prof, shape=disc, t=2):
    """prof: list of (z0, z1, r, mode) mode in {'solid','ring'}"""
    for z0, z1, r, mode in prof:
        cells = shape(r) if mode == "solid" else (shape(r) - shape(r - t))
        put(V, cells, z0, z1)


# ---------------------------------------------------------------- designs (index -> fn(round))
D = {}


def d(i):
    def w(f):
        D[i] = f; return f
    return w


@d(0)   # tall bottle narrow neck
def _(r):
    V = set()
    revolve(V, [(0, 1, 5, "solid"), (1, 10, 5, "ring"), (10, 11, 5, "solid"), (11, 12, 4, "solid"),
                (12, 13, 3, "solid"), (13, 18, 2, "ring"), (18, 19, 2.3, "solid")])
    return V


@d(1)   # square pedestal table
def _(r):
    V = set()
    box(V, 4, 16, 4, 16, 0, 1); box(V, 6, 14, 6, 14, 1, 2)
    box(V, 8, 12, 8, 12, 2, 10, hollow=True)
    if r < 2:
        box(V, 6, 14, 6, 14, 9, 10) if r else None
        box(V, 6, 14, 6, 14, 10, 11) if not r else box(V, 5, 15, 5, 15, 10, 11)
        box(V, 2, 18, 2, 18, 11, 13) if not r else box(V, 3, 17, 3, 17, 11, 13)
    else:
        box(V, 6, 14, 6, 14, 9, 10); box(V, 4, 16, 4, 16, 10, 11); box(V, 3, 17, 3, 17, 11, 13)
    rm(V, 4, 16, 4, 16, 11, 12) if r >= 1 else None
    box(V, 5, 15, 5, 15, 11, 12, hollow=True) if r >= 1 else None
    return V


@d(2)   # toy car segmented top
def _(r):
    V = set()
    car(V, L=14, W=6, y0=3, x0=7, body=1, cab=(3, 11), cab_h=1, tiers=1)
    for y in range(3 + 4, 3 + 11, 3):    # segmented roof ridges
        box(V, 8, 12, y, y + 2, 4, 5)
    return V


@d(3)   # rectangular flat table
def _(r):
    V = set(); table(V, 10, 16, 8, leg=2, top=1); return V


@d(4)   # square table four sturdy legs
def _(r):
    V = set(); table(V, 14, 14, 10, leg=3, top=2); return V


@d(5)   # grand piano, three legs, stepped top
def _(r):
    V = set()
    for x, y in ([(4, 3), (13, 3), (8, 14)] if r < 2 else [(4, 0), (13, 0), (8, 14)]):
        box(V, x, x + 2, y, y + 2, 0, 5)
    # body: wide at keyboard (y=2..6), tapering toward tail (y high)
    for z in (5, 6, 7):
        for y in range(2, 18):
            w = 14 if y < 8 else max(6, 14 - (y - 8))
            box(V, 3, 3 + w, y, y + 1, z, z + 1, hollow=False) if z == 5 or (r and z == 6) else None
            if z > 5:
                for x in (3, 3 + w - 1):
                    V.add((x, y, z))
                if y in (2, 17):
                    box(V, 3, 3 + w, y, y + 1, z, z + 1)
    box(V, 3, 17, 0, 2, 5, 6)              # keyboard shelf
    for x in range(3, 17, 2):
        V.add((x, 0, 6))                   # black keys
    # stepped lid
    for y in range(2, 18):
        w = 14 if y < 8 else max(6, 14 - (y - 8))
        box(V, 3, 3 + w, y, y + 1, 8, 9)
    box(V, 5, 13, 6, 14, 9, 10); box(V, 7, 11, 9, 12, 10, 11)
    return V


@d(6)   # low table square top two side panels
def _(r):
    V = set()
    box(V, 2, 4, 2, 18, 0, 6); box(V, 16, 18, 2, 18, 0, 6)
    for z in range(2, 4):
        rm(V, 2, 4, 6, 14, z, z + 1); rm(V, 16, 18, 6, 14, z, z + 1)    # cut-outs in panels
    box(V, 4, 16, 2, 18, 1, 2) if r >= 1 else None       # low shelf
    box(V, 1, 19, 1, 19, 6, 8)
    return V


@d(7)   # vessel elongated flat base, small
def _(r):
    V = set()
    if r == 0:
        vessel(V, L=14, B=4, x0=8, y0=3, deck_z=1, tiers=((4, 9, 1),), step_sides=False)
        box(V, 9, 11, 12, 14, 2, 3)
    else:
        box(V, 8, 12, 3, 17, 0, 1); box(V, 9, 11, 17, 18, 0, 1)        # flat hull + pointed bow
        box(V, 8, 12, 3, 17, 1, 2, hollow=True); V.add((9, 17, 1)); V.add((10, 17, 1))
        box(V, 8, 12, 4, 9, 2, 3)                                    # deckhouse
        box(V, 9, 11, 5, 7, 3, 4)                                    # wheelhouse
    return V


@d(8)   # train layered angular, small
def _(r):
    V = set(); train(V, L=14, W=4, H=3, x0=8, tiers=2)
    box(V, 9, 11, 11, 13, 7 - r, 9)          # chimney
    return V


@d(9)   # round table cylindrical legs
def _(r):
    V = set()
    for cx, cy in [(6, 6), (14, 6), (6, 14), (14, 14)]:
        put(V, disc(1.2, cx, cy), 0, 8)
    put(V, ring(6.5, 1.5) , 7, 8)
    put(V, disc(8.5), 8, 10)
    return V


@d(10)  # rectangular table open compartments beneath
def _(r):
    V = set()
    box(V, 2, 18, 4, 16, 0, 1)
    for x in (2, 7, 12, 16):
        box(V, x, x + 2, 4, 16, 1, 8)
        rm(V, x, x + 2, 6, 14, 3, 6)
    box(V, 2, 18, 4, 16, 4, 5) if r == 0 else (box(V, 2, 18, 4, 16, 6, 7) if r == 1 else None)
    box(V, 1, 19, 3, 17, 8, 9) if r < 3 else box(V, 2, 18, 4, 16, 8, 9)
    return V


@d(11)  # angular sectional (L-shaped)
def _(r):
    V = set()
    box(V, 2, 16, 4, 9, 0, 2)      # long seat
    box(V, 11, 16, 9, 16, 0, 2)    # chaise leg of L
    box(V, 2, 16, 4, 6, 2, 5)      # back along long side
    box(V, 14, 16, 6, 16, 2, 5)    # back along short side (connected corner)
    box(V, 2, 4, 6, 9, 2, 3)       # arm
    return V


@d(12)  # vessel central vertical extension
def _(r):
    V = set()
    vessel(V, L=18, B=8, x0=6, y0=1, deck_z=3, tiers=((4, 14, 2), (6, 12, 2)))
    box(V, 9, 11, 8, 10, 8, 17)         # mast / funnel
    box(V, 7, 13, 8, 10, 14, 15)        # yard arm
    return V


@d(13)  # jar rounded body layered surface
def _(r):
    V = set()
    prof = [(0, 1, 4, "solid"), (1, 2, 5, "ring"), (2, 4, 6.5, "ring"), (4, 9, 7.5, "ring"),
            (9, 11, 6.5, "ring"), (11, 12, 5, "ring"), (12, 13, 4, "ring"), (13, 15, 4.5, "ring")]
    if r < 2:
        revolve(V, prof)
        put(V, disc(6.5) - disc(5.5), 6, 7)   # band (layered surface)
        put(V, disc(8) - disc(6.5), 6, 7)
    else:   # layered surface = alternating bulge/recess rings, no outward overhang beyond 1 cell
        prof = [(0, 1, 4, "solid"), (1, 2, 5, "ring"), (2, 3, 6, "ring"), (3, 4, 7, "ring")]
        prof += [(z, z + 1, 7.5 if z % 2 else 7, "ring") for z in range(4, 10)]
        prof += [(10, 11, 6.5, "ring"), (11, 12, 5.5, "ring"), (12, 13, 4.5, "ring"), (13, 15, 4.5, "ring")]
        revolve(V, prof)
    return V


@d(14)  # tiny rectangular vessel tiered edge
def _(r):
    V = set()
    if r == 0:
        box(V, 8, 12, 5, 15, 0, 1); box(V, 7, 13, 4, 16, 1, 2, hollow=True); box(V, 9, 11, 8, 12, 2, 3)
    else:
        box(V, 8, 12, 5, 15, 0, 1); box(V, 7, 13, 4, 16, 1, 2); box(V, 7, 13, 4, 16, 2, 3, hollow=True)
        box(V, 9, 11, 8, 12, 2, 4)
    return V


@d(15)  # compact car tiered layers
def _(r):
    V = set(); car(V, L=16, W=8, y0=2, x0=6, body=2, cab=(4, 11), cab_h=2, tiers=2); return V


@d(16)  # compact chair open-frame legs
def _(r):
    V = set(); chair(V, S=8, D=8, seat=6, back=8, leg=1, back_style="slats", stretch=True); return V


@d(17)  # compact vehicle geometric tiered
def _(r):
    V = set(); car(V, L=16, W=8, y0=2, x0=6, body=2, cab=(3, 12), cab_h=2, tiers=3)
    box(V, 7, 13, 15, 17, 3, 4)       # hood scoop
    return V


@d(18)  # simple table flat rect top
def _(r):
    V = set(); table(V, 12, 18, 9, leg=2, top=2, apron=False, inset=1); return V


@d(19)  # tall hexagonal jar vertical panels
def _(r):
    V = set()
    put(V, hexa(6.5), 0, 1)
    put(V, hexa(6.5) - hexa(4.5), 1, 15)
    for z in range(2, 14):
        for x, y in list(hexa(6.5) - hexa(5.5)):
            if (x + y) % 4 == 0 and z % 4 != 1:
                V.discard((x, y, z)) if (x, y) in hexa(6.5) - hexa(5.6) and False else None
    put(V, hexa(5.5) - hexa(3.5), 15, 16)
    put(V, hexa(4.5) - hexa(2.5), 16, 18)
    return V


@d(20)  # compact car angular roof
def _(r):
    V = set(); car(V, L=16, W=8, y0=2, x0=6, body=2, cab=(4, 12), cab_h=2, tiers=2)
    rm(V, 6, 14, 2, 3, 2, 3); return V


@d(21)  # guitar
def _(r):
    V = set()
    box(V, 6, 14, 1, 9, 0, 1); box(V, 7, 13, 0, 10, 0, 1)
    box(V, 6, 14, 1, 9, 1, 2); rm(V, 9, 11, 4, 6, 1, 2)
    box(V, 9, 11, 9, 18, 0, 1); box(V, 8, 12, 18, 20, 0, 1)
    box(V, 9, 11, 9, 17, 1, 2)
    return V


@d(22)  # long bus raised roof section
def _(r):
    V = set(); bus(V, L=20, W=6, H=4, raised=(5, 15), x0=7); return V


@d(23)  # chair high backrest
def _(r):
    V = set(); chair(V, S=8, D=8, seat=6, back=11, leg=2, back_style="solid", stretch=False); return V


@d(24)  # simple rect table four sturdy legs
def _(r):
    V = set(); table(V, 10, 14, 8, leg=2, top=2 - min(r, 1)); return V


@d(25)  # angular table unique leg placement: trestle/X legs offset
def _(r):
    V = set()
    box(V, 3, 17, 3, 7, 0, 1); box(V, 3, 17, 13, 17, 0, 1)          # feet bars
    for k, z in enumerate(range(1, 7)):                               # slanted legs leaning inward
        box(V, 9 - 0, 11, 4 + k // 2, 6 + k // 2, z, z + 1); box(V, 9, 11, 14 - k // 2, 16 - k // 2, z, z + 1)
    box(V, 9, 11, 7, 13, 6, 7)
    box(V, 2, 18, 4, 16, 7, 8)
    box(V, 4, 16, 2, 18, 8, 9) if not r else box(V, 3, 17, 4, 16, 8, 9)
    return V


@d(26)  # chair stepped angular backrest
def _(r):
    V = set(); chair(V, S=10, D=9, seat=6, back=10, leg=2, x0=5, y0=5, back_style="step"); return V


@d(27)  # bus elongated blocky
def _(r):
    V = set(); bus(V, L=18, W=7, H=5, x0=6, raised=(1, 17) if r else None); box(V, 8, 11, 12, 16, 6 + min(r, 1), 7 + min(r, 1)); return V


@d(28)  # pot rounded top structured base
def _(r):
    V = set()
    box(V, 4, 16, 4, 16, 0, 1); box(V, 4, 16, 4, 16, 1, 3, hollow=True)     # square plinth
    box(V, 5, 15, 5, 15, 3, 4) if r < 2 else box(V, 4, 16, 4, 16, 3, 4)
    revolve(V, [(4, 6, 5.5, "ring"), (6, 10, 6.5, "ring"), (10, 12, 7, "ring"), (12, 13, 6, "ring"),
                (13, 14, 7.5, "ring")] if r == 0 else [(4, 6, 5.5, "ring"), (6, 10, 6.5, "ring"), (10, 12, 7, "ring"),
                (12, 13, 6.5, "ring"), (13, 14, 7, "ring")])
    put(V, disc(5.5) - disc(4), 13, 14)
    return V


@d(29)  # chair straight backrest cubic base
def _(r):
    V = set()
    box(V, 5, 15, 5, 15, 0, 6, hollow=True); box(V, 5, 15, 5, 15, 6, 7)
    box(V, 5, 15, 13, 15, 7, 17)
    for z in (8, 10, 12, 14):
        rm(V, 7, 13, 13, 15, z, z + 1) if False else None
    rm(V, 7, 13, 5, 6, 1, 4); rm(V, 7, 13, 14, 15, 1, 4)       # openings in cube sides
    return V


@d(30)  # chair blocky back & seat, straight legs
def _(r):
    V = set()
    chair(V, S=10, D=10, seat=7, back=10, leg=2, x0=5, y0=5, back_style="solid", stretch=False)
    box(V, 5, 15, 5, 15, 7, 9)
    box(V, 5, 15, 12, 15, 9, 17)
    return V


@d(31)  # square table four sturdy legs (variant)
def _(r):
    V = set(); table(V, 12, 12, 9, leg=3, top=2, apron=False)
    box(V, 4, 16, 4, 16, 3, 4, hollow=True)    # low stretcher ring
    return V


@d(32)  # simple chair segmented backrest
def _(r):
    V = set(); chair(V, S=10, D=10, seat=7, back=10, leg=2, x0=5, y0=5, back_style="slats"); return V


@d(33)  # chair articulated backrest distinct supports
def _(r):
    V = set()
    chair(V, S=10, D=10, seat=7, back=0, leg=2, x0=5, y0=5, back_style="solid")
    for x in (5, 13):
        box(V, x, x + 2, 13, 15, 8, 17)                 # supports
    box(V, 7, 13, 13, 15, 10, 13); box(V, 7, 13, 14, 16, 13, 16) if False else box(V, 7, 13, 13, 15, 14, 16)
    box(V, 5, 15, 13, 15, 16, 17)
    for x in (5, 13):
        box(V, x, x + 2, 5, 13 + 2 * min(r, 1), 10, 11)                 # arm supports
        box(V, x, x + 2, 5, 7, 8, 10)
    return V


@d(34)  # elongated vessel stepped sides
def _(r):
    V = set()
    if r == 0:
        vessel(V, L=16, B=6, x0=7, y0=2, deck_z=3, tiers=((5, 11, 1),)); return V
    box(V, 9, 11, 4, 16, 0, 1); box(V, 8, 12, 3, 17, 1, 2); box(V, 7, 13, 2, 18, 2, 3)   # stepped sides
    box(V, 7, 13, 2, 18, 3, 4, hollow=True); box(V, 8, 12, 7, 13, 3, 5); return V


@d(35)  # compact sofa straight back
def _(r):
    V = set(); sofa(V, L=16, D=8, seat=3, back=6, arm=2, arm_h=5, x0=2, y0=6); return V


@d(36)  # bookshelf three shelves
def _(r):
    V = set()
    box(V, 3, 17, 6, 10, 0, 1)
    box(V, 3, 5, 6, 10, 1, 17); box(V, 15, 17, 6, 10, 1, 17)    # sides
    box(V, 5, 15, 9, 10, 1, 17)                                  # back panel
    for z in (5, 9, 13):
        box(V, 3, 17, 6, 10, z, z + 1)
    box(V, 3, 17, 6, 10, 17, 18)
    for x in (6, 8, 9, 11):
        box(V, x, x + 1, 7, 9, 6, 8 if x % 2 else 9)              # books
    return V


@d(37)  # compact table layered top thick legs
def _(r):
    V = set(); table(V, 12, 12, 8, leg=3, top=2, apron=False)
    box(V, 3, 17, 3, 17, 8, 9); return V


@d(38)  # chair square seat elongated backrest
def _(r):
    V = set(); chair(V, S=8, D=8, seat=5, back=11, leg=1, back_style="vslats", stretch=True); return V


@d(39)  # compact chair distinctive geometric
def _(r):
    V = set()
    box(V, 6, 14, 6, 14, 0, 1)
    if r == 0:
        box(V, 9, 11, 9, 11, 1, 5); box(V, 7, 13, 7, 13, 4, 5)
    elif r == 1:
        box(V, 8, 12, 8, 12, 1, 3, hollow=True); box(V, 6, 14, 6, 14, 3, 4); box(V, 5, 15, 5, 15, 4, 5, hollow=True)
    else:
        box(V, 6, 14, 6, 14, 1, 4, hollow=True); box(V, 5, 15, 5, 15, 4, 5)
    box(V, 5, 15, 5, 15, 5, 7)                     # seat
    box(V, 5, 15, 13, 15, 7, 13); rm(V, 7, 13, 13, 15, 8, 11) if r == 0 else None   # framed back
    box(V, 5, 7, 5, 13, 7, 9); box(V, 13, 15, 5, 13, 7, 9)   # geometric arms
    return V


@d(40)  # compact chair straight back simple legs
def _(r):
    V = set(); chair(V, S=8, D=8, seat=6, back=8, leg=1, back_style="solid", stretch=True); return V


@d(41)  # simple chair angular (raked) backrest
def _(r):
    V = set()
    chair(V, S=8, D=8, seat=6, back=0, leg=2, back_style="solid", stretch=False)
    for k, z in enumerate(range(7, 15)):
        y = 12 + k // 3
        box(V, 6, 14, y, y + 2, z, z + 1)
    return V


@d(42)  # bench flat seat + backrest
def _(r):
    V = set()
    b = 11 if r < 2 else 10
    for y in ((1, 17) if r == 0 else (1, 9, 17)):
        box(V, 6, 8, y, y + 2, 0, 3); box(V, b, b + 2, y, y + 2, 0, 3)
    box(V, 6, b + 2, 1, 19, 3, 4)
    box(V, b, b + 2, 1, 3, 4, 8); box(V, b, b + 2, 17, 19, 4, 8)
    box(V, b, b + 2, 1, 19, 6, 7); box(V, b, b + 2, 1, 19, 8, 9)
    return V


@d(43)  # rectangular train layered
def _(r):
    V = set(); train(V, L=16, W=5, H=3, x0=8, tiers=2, e=1 if r < 2 else 0); box(V, 9, 12, 2, 5, 6, 8); return V


@d(44)  # jar geometric upright (octagon-ish stepped)
def _(r):
    V = set()
    sq = lambda s: {(x, y) for x in range(10 - s, 10 + s) for y in range(10 - s, 10 + s)
                    if abs(x + .5 - 10) + abs(y + .5 - 10) <= s * 1.45}
    put(V, sq(5), 0, 1)
    for z0, z1, s in [(1, 3, 5), (3, 12, 6), (12, 14, 5), (14, 15, 4)]:
        put(V, sq(s) - sq(s - 2), z0, z1)
    put(V, sq(4) - sq(2), 15, 17); put(V, sq(5) - sq(3), 17, 18) if r < 2 else put(V, sq(4) - sq(2), 17, 18)
    return V


@d(45)  # table flat top robust legs
def _(r):
    V = set(); table(V, 10, 12, 7, leg=3, top=1); return V


@d(46)  # compact sofa straight back cubic
def _(r):
    V = set(); sofa(V, L=12, D=8, seat=3, back=5, arm=2, arm_h=5, x0=4, y0=6, legs=False); return V


@d(47)  # upright piano layered top vertical slats
def _(r):
    V = set()
    box(V, 3, 17, 8, 14, 0, 1)                              # plinth
    box(V, 3, 17, 10, 14, 1, 12, hollow=True)               # cabinet
    for x in range(5, 15, 2):
        box(V, x, x + 1, 9, 10, 1, 5)                       # vertical slats (front lower panel)
    box(V, 3, 17, 8, 10, 1, 5, hollow=False) if r >= 1 else None
    box(V, 3, 5, 6, 10, 1, 7); box(V, 15, 17, 6, 10, 1, 7)  # cheeks
    box(V, 5, 15, 6, 10, 5, 6)                              # keybed
    for x in range(5, 15, 2):
        V.add((x, 8, 6))                                    # black keys
    box(V, 3, 17, 10, 14, 12, 13); box(V, 2, 18, 9, 15, 13, 14)   # layered top
    return V


@d(48)  # sofa symmetrical armrests
def _(r):
    V = set(); sofa(V, L=16, D=9, seat=3, back=5, arm=3, arm_h=6, x0=2, y0=5); return V


@d(49)  # compact car layered angular profile
def _(r):
    V = set(); car(V, L=16, W=8, y0=2, x0=6, body=2, cab=(5, 12), cab_h=2, tiers=2)
    box(V, 7, 13, 2, 4, 3, 4)       # spoiler base
    return V


ALLOWED = P.SIZES


def tile2(V):
    """Support-aware greedy tiling: unsupported cells are covered first by the largest brick that also
    covers a supported cell (voxel below), so bridges/overhangs are anchored; orientation alternates by layer."""
    lines = []
    for z in sorted({v[2] for v in V}):
        free = {(x, y) for x, y, zz in V if zz == z}
        sup = {(x, y) for (x, y) in free if z == 0 or (x, y, z - 1) in V}
        order = ALLOWED if z % 2 == 0 else [(w, h) for h, w in ALLOWED]
        key = lambda p: (p, p[::-1])[z % 2]
        uns = sorted(free - sup, key=key)

        def cands(x, y):
            for h, w in order:
                for dx in range(h):
                    for dy in range(w):
                        cells = [(x - dx + i, y - dy + j) for i in range(h) for j in range(w)]
                        if all(c in free for c in cells):
                            yield h * w, sum(c in sup for c in cells), (x - dx, y - dy, h, w), cells
        for x, y in uns:
            if (x, y) not in free:
                continue
            cs = list(cands(x, y))
            anch = [c for c in cs if c[1] > 0]
            best = max(anch or cs, key=lambda c: (c[0] + 2 * min(c[1], 4), c[1]))
            free -= set(best[3]); bx, by, h, w = best[2]; lines.append((z, f"{h}x{w} ({bx},{by},{z})"))
        for x, y in sorted(free, key=key):
            if (x, y) not in free:
                continue
            for h, w in order:
                cells = [(x + i, y + j) for i in range(h) for j in range(w)]
                if all(c in free for c in cells):
                    free -= set(cells); lines.append((z, f"{h}x{w} ({x},{y},{z})")); break
    return chr(10).join(l for _, l in lines)


RND = {}


def tile3(V):
    """tile2 but unsupported cells are handled most-constrained-first (fewest anchored brick options)."""
    lines = []
    for z in sorted({v[2] for v in V}):
        free = {(x, y) for x, y, zz in V if zz == z}
        sup = {(x, y) for (x, y) in free if z == 0 or (x, y, z - 1) in V}
        order = ALLOWED if z % 2 == 0 else [(w, h) for h, w in ALLOWED]
        key = lambda p: (p, p[::-1])[z % 2]

        def cands(x, y):
            out = []
            for h, w in order:
                for dx in range(h):
                    for dy in range(w):
                        cells = [(x - dx + i, y - dy + j) for i in range(h) for j in range(w)]
                        if all(c in free for c in cells):
                            out.append((h * w, sum(c in sup for c in cells), (x - dx, y - dy, h, w), cells))
            return out
        uns = set(free - sup)
        while uns:
            info = {c: cands(*c) for c in uns}
            na = {c: sum(1 for k in v if k[1] > 0) for c, v in info.items()}
            pos = [c for c in uns if na[c] > 0]
            c = min(pos, key=lambda c: (na[c], key(c))) if pos else min(uns, key=key)
            cs = info[c]; anch = [k for k in cs if k[1] > 0]
            best = max(anch or cs, key=lambda k: (k[0] + 2 * min(k[1], 4), k[1]))
            free -= set(best[3]); uns -= set(best[3]); bx, by, h, w = best[2]
            lines.append(f"{h}x{w} ({bx},{by},{z})")
        for x, y in sorted(free, key=key):
            if (x, y) not in free:
                continue
            for h, w in order:
                cells = [(x + i, y + j) for i in range(h) for j in range(w)]
                if all(c in free for c in cells):
                    free -= set(cells); lines.append(f"{h}x{w} ({x},{y},{z})"); break
    return chr(10).join(lines)


T3 = {10, 13, 28, 33, 39, 42, 43, 44}


def build(i, rnd):
    PREF.clear()
    return (tile3 if i in T3 and rnd > 0 else tile2)(D[i](rnd))


def main():
    prompts = [json.loads(l) for l in open(ROOT / "data/opus_gen/prompts_2.jsonl", encoding="utf-8")]
    only = [int(a.split(":")[0]) for a in sys.argv[1:]]
    RND.update({int(a.split(":")[0]): int(a.split(":")[1]) for a in sys.argv[1:] if ":" in a})
    (HERE / "renders").mkdir(exist_ok=True)
    res = {}
    for i, p in enumerate(prompts):
        if only and i not in only:
            continue
        t0 = time.time()
        text = build(i, RND.get(i, 0))
        r = check(text, use_gurobi=True, gurobi_kwargs={"time_limit": 400}).to_dict()
        try:
            f = feats(text)
        except Exception:
            f = {"n_bricks": 0, "n_types": 0, "max_type_share": 1, "fill": 1}
        P.render_png(text, HERE / "renders" / f"{i:02d}_{p['structure_id'][:8]}.png")
        res[i] = {"rnd": RND.get(i, 0), "ok": r["ok"], "msg": r["message"][:150], **f, "keep": keep(f), "text": text}
        print(i, p["bucket"], p["n_bricks"], "->", f["n_bricks"], r["ok"], keep(f), round(f["fill"], 2), r["message"][:100],
              round(time.time() - t0, 1), flush=True)
    old = json.load(open(HERE / "_draft.json")) if (HERE / "_draft.json").exists() else {}
    old.update({str(k): v for k, v in res.items()})
    json.dump(old, open(HERE / "_draft.json", "w"))


if __name__ == "__main__":
    main()
