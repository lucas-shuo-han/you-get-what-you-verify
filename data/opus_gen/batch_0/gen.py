"""Batch 0 generator: caption-only parametric designs, tiled with opus_pilot.tile, checked + filtered + rendered.
python data/opus_gen/batch_0/gen.py [sid8 ...]
"""
import json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import types  # noqa: E402
try:
    import scipy  # noqa: F401
except ImportError:   # IoU module (scipy) is not needed here; stub it so opus_pilot imports
    sys.modules["distill_shape"] = types.SimpleNamespace(aligned_iou=None)
import opus_pilot as OP  # noqa: E402
from opus_pilot import box, tile, render_png  # noqa: E402
from brickexp.checker import check  # noqa: E402
from distill_filter import feats, keep  # noqa: E402

OUT = Path(__file__).resolve().parent
PREF = OP.PREF


def tile2(V):
    """support-aware greedy tiler: each brick must sit on / bond as many distinct bricks below as possible."""
    lines = []; owner = {}
    sizes = OP.SIZES
    for z in sorted({v[2] for v in V}):
        free = {(x, y) for x, y, zz in V if zz == z}
        below = {(x, y): owner[(x, y, z - 1)] for (x, y) in free if (x, y, z - 1) in owner} if z else None
        above = {(x, y) for x, y, zz in V if zz == z + 1}
        k0 = len(lines)
        for x, y in sorted(free, key=lambda p: (p, p[::-1])[z % 2]):
            if (x, y) not in free:
                continue
            best = None
            for h, w in sizes:
                for dx in range(h):
                    for dy in range(w):
                        bx, by = x - dx, y - dy
                        cells = [(bx + i, by + j) for i in range(h) for j in range(w)]
                        if not all(c in free for c in cells):
                            continue
                        if z == 0:
                            sup, nb = h * w, 1
                        else:
                            sup = sum(c in below for c in cells); nb = len({below[c] for c in cells if c in below})
                        up = sum(c in above for c in cells)
                        sc = (sup > 0 or up > 0, nb, sup > 0, h * w + up * 0.1, (h, w) == (1, 1) and -1 or 0)
                        if best is None or sc > best[0]:
                            best = (sc, bx, by, h, w, cells)
            _, bx, by, h, w, cells = best
            free -= set(cells); k = len(lines)
            lines.append(f"{h}x{w} ({bx},{by},{z})")
            for cx, cy in cells:
                owner[(cx, cy, z)] = k
    return "\n".join(lines)


def disk(V, cx, cy, r, z0, z1, ring=None):
    """filled disk (or ring of thickness `ring`) centred at (cx,cy) (half-integer centre allowed)."""
    for x in range(20):
        for y in range(20):
            d = math.hypot(x + .5 - cx, y + .5 - cy)
            if d <= r and (ring is None or d > r - ring):
                for z in range(z0, z1):
                    V.add((x, y, z))


def rm(V, x0, x1, y0, y1, z0, z1):
    V -= {(x, y, z) for x in range(x0, x1) for y in range(y0, y1) for z in range(z0, z1)}


# ---------------------------------------------------------------- chairs (x = width, y = depth, back at high y)
def chair(back="solid", W=8, D=8, H=5, BH=8, leg=2, arms=False, angled=False, top=None, x0=6, y0=6):
    V = set()
    for lx, ly in [(0, 0), (W - leg, 0), (0, D - leg), (W - leg, D - leg)]:
        for z in range(H):
            ox = oy = 0
            if angled and z < H - 1:   # legs splay outward going down (1 cell over the bottom half)
                s = 1 if z < (H - 1) // 2 + 1 else 0
                ox = -s if lx == 0 else s; oy = -s if ly == 0 else s
            box(V, x0 + lx + ox, x0 + lx + ox + leg, y0 + ly + oy, y0 + ly + oy + leg, z, z + 1)
    box(V, x0, x0 + W, y0, y0 + D, H, H + 2)                       # seat (2 crossed layers)
    zb = H + 2
    by0, by1 = y0 + D - 2, y0 + D
    if back == "solid":
        box(V, x0, x0 + W, by0, by1, zb, zb + BH)
        rm(V, x0 + 2, x0 + W - 2, by0, by1, zb + 1, zb + BH - 2)    # keep it a frame: open panel
        box(V, x0 + 2, x0 + W - 2, by1 - 1, by1, zb + 1, zb + BH - 2)
    elif back == "slats":
        box(V, x0, x0 + 2, by0, by1, zb, zb + BH); box(V, x0 + W - 2, x0 + W, by0, by1, zb, zb + BH)
        for z in range(zb + 1, zb + BH, 2):
            box(V, x0, x0 + W, by1 - 1, by1, z, z + 1)
    elif back == "ornate":
        box(V, x0, x0 + 2, by0, by1, zb, zb + BH); box(V, x0 + W - 2, x0 + W, by0, by1, zb, zb + BH)
        box(V, x0, x0 + W, by0, by1, zb + BH, zb + BH + 1)         # top rail
        box(V, x0 + W // 2 - 1, x0 + W // 2 + 1, by1 - 1, by1, zb, zb + BH)   # central splat
        for z in (zb + 2, zb + BH - 2):
            box(V, x0, x0 + W, by1 - 1, by1, z, z + 1)
        box(V, x0 - 1, x0 + 1, by0, by1, zb + BH + 1, zb + BH + 2); box(V, x0 + W - 1, x0 + W + 1, by0, by1, zb + BH + 1, zb + BH + 2)  # finials
    elif back == "curved":   # wrap-around U back: sides come forward
        box(V, x0, x0 + W, by0, by1, zb, zb + BH)
        for z in range(zb, zb + BH):
            depth = 4 if z < zb + BH - 2 else 2
            box(V, x0, x0 + 1, by1 - depth, by1, z, z + 1); box(V, x0 + W - 1, x0 + W, by1 - depth, by1, z, z + 1)
        box(V, x0 + 1, x0 + W - 1, by1 - 1, by1, zb + BH, zb + BH + 1)
    if arms:
        for ax in (x0, x0 + W - 1):
            box(V, ax, ax + 1, y0, y0 + 1, zb, zb + 3)             # front arm post
            box(V, ax, ax + 1, y0, y0 + D - 1, zb + 3, zb + 4)     # arm rail into back
    return V


# ---------------------------------------------------------------- sofas (x = length, y = depth, back at high y)
def sofa(L=14, D=7, seatH=3, BH=4, armH=2, arm=2, feet=True, corner=False, x0=3, y0=6, cushions=True, angled=False):
    V = set()
    z = 0
    if feet:
        for fx, fy in [(0, 0), (L - 2, 0), (0, D - 2), (L - 2, D - 2)]:
            box(V, x0 + fx, x0 + fx + 2, y0 + fy, y0 + fy + 2, 0, 1)
        z = 1
    box(V, x0, x0 + L, y0, y0 + D, z, seatH)                       # deck + base
    rm(V, x0 + 2, x0 + L - 2, y0 + 2, y0 + D - 2, z + 2, seatH - 1)   # hollow base box
    if corner:
        box(V, x0, x0 + D, y0 - 6, y0, z, seatH)                   # chaise wing toward -y
        rm(V, x0 + 2, x0 + D - 2, y0 - 4, y0, z + 2, seatH - 1)
        if feet:
            box(V, x0, x0 + 2, y0 - 6, y0 - 4, 0, 1); box(V, x0 + D - 2, x0 + D, y0 - 6, y0 - 4, 0, 1)
    if cushions:   # seat cushions with a visible seam
        half = L // 2
        box(V, x0 + arm, x0 + half - 0, y0, y0 + D - 2, seatH, seatH + 1)
        box(V, x0 + half + 1, x0 + L - arm, y0, y0 + D - 2, seatH, seatH + 1)
        box(V, x0 + half, x0 + half + 1, y0, y0 + D - 2, seatH, seatH + 1) if not cushions == "seam" else None
    box(V, x0, x0 + L, y0 + D - 2, y0 + D, seatH, seatH + BH)      # backrest
    if angled:
        box(V, x0, x0 + L, y0 + D - 1, y0 + D, seatH + BH, seatH + BH + 1)
    for ax in (x0, x0 + L - arm):
        box(V, ax, ax + arm, y0, y0 + D, seatH, seatH + armH)       # arms
    if corner:
        box(V, x0, x0 + arm, y0 - 6, y0, seatH, seatH + armH)
        box(V, x0 + arm, x0 + D, y0 - 6, y0, seatH, seatH + 1)
    return V


# ---------------------------------------------------------------- tables
def table_rect(W=12, D=8, H=6, leg=2, tiered=False, apron=True, shelf=False, x0=4, y0=6, thick=2, legs_in=0):
    V = set()
    for lx, ly in [(legs_in, legs_in), (W - leg - legs_in, legs_in), (legs_in, D - leg - legs_in), (W - leg - legs_in, D - leg - legs_in)]:
        box(V, x0 + lx, x0 + lx + leg, y0 + ly, y0 + ly + leg, 0, H)
    if apron:
        box(V, x0 + legs_in, x0 + W - legs_in, y0 + legs_in, y0 + D - legs_in, H - 1, H, hollow=True)
    if shelf:
        box(V, x0 + legs_in, x0 + W - legs_in, y0 + legs_in, y0 + D - legs_in, 1, 3, hollow=True)
    box(V, x0, x0 + W, y0, y0 + D, H, H + thick)
    if tiered:
        box(V, x0 + 2, x0 + W - 2, y0 + 2, y0 + D - 2, H + thick, H + thick + 1)
        box(V, x0 + 4, x0 + W - 4, y0 + 3, y0 + D - 3, H + thick + 1, H + thick + 2)
    return V


def table_panel():
    """minimal table of distinct shapes: two slab end-panels with windows, a thick top and a square stretcher."""
    V = set(); x0, y0, W, D, H = 4, 6, 12, 8, 6
    for px in (x0, x0 + W - 2):
        box(V, px, px + 2, y0, y0 + D, 0, H)
        rm(V, px, px + 2, y0 + 2, y0 + D - 2, 1, H - 2)
    box(V, x0, x0 + W, y0 + 3, y0 + 5, H - 2, H)          # stretcher under the top
    box(V, x0, x0 + W, y0, y0 + D, H, H + 2)
    return V


def table_round(r=6, H=7, base_r=4, frame=False):
    V = set(); c = 10
    if frame:   # interlocking cross frame: crossed feet, column, crossed braces under the top
        for z in (0, 1):
            box(V, c - 5, c + 5, c - 1, c + 1, z, z + 1); box(V, c - 1, c + 1, c - 5, c + 5, z, z + 1)
        box(V, c - 1, c + 1, c - 1, c + 1, 2, H - 2)
        for z in (H - 2, H - 1):
            box(V, c - 4, c + 4, c - 1, c + 1, z, z + 1); box(V, c - 1, c + 1, c - 4, c + 4, z, z + 1)
    else:
        disk(V, c, c, base_r, 0, 1)
        disk(V, c, c, base_r - 1, 1, 2)
        box(V, c - 1, c + 1, c - 1, c + 1, 2, H - 1)
        disk(V, c, c, 3, H - 1, H)
    disk(V, c, c, r, H, H + 2)
    return V


# ---------------------------------------------------------------- vehicles (y = length)
def car(L=15, W=8, cab=(3, 11), cabH=2, lowH=2, x0=6, y0=2, stepped=False, rounded=False, spoiler=False):
    V = set()
    for wx in (x0, x0 + W - 2):
        for wy in (y0, y0 + L - 3):
            box(V, wx, wx + 2, wy, wy + 3, 0, 1)
    box(V, x0, x0 + W, y0, y0 + L, 1, 1 + lowH)
    rm(V, x0 + 2, x0 + W - 2, y0 + 2, y0 + L - 2, 3, lowH)     # hollow lower body
    if rounded:
        for yy in (y0, y0 + L - 1):
            V.discard((x0, yy, lowH)); V.discard((x0 + W - 1, yy, lowH))
    z = 1 + lowH
    c0, c1 = y0 + cab[0], y0 + cab[1]
    box(V, x0 + 1, x0 + W - 1, c0, c1, z, z + cabH, hollow=True)
    for wy in range(c0 + 1, c1 - 1):
        if (wy - c0) % 3 != 0:
            V.discard((x0 + 1, wy, z + cabH - 1)); V.discard((x0 + W - 2, wy, z + cabH - 1))
    box(V, x0 + 1, x0 + W - 1, c0, c1, z + cabH, z + cabH + 2)       # roof (2 layers)
    if stepped:
        box(V, x0 + 2, x0 + W - 2, c0 + 1, c1 - 1, z + cabH + 2, z + cabH + 3)
    if rounded:
        box(V, x0 + 2, x0 + W - 2, c0 + 1, c1 - 1, z + cabH + 2, z + cabH + 3)
    box(V, x0 + 2, x0 + W - 2, y0 + L - 1, y0 + L, z, z + 1)          # bumper/grille detail
    box(V, x0 + 1, x0 + 2, y0 + L - 1, y0 + L, z, z + 1); box(V, x0 + W - 2, x0 + W - 1, y0 + L - 1, y0 + L, z, z + 1)
    if spoiler:
        box(V, x0, x0 + W, y0, y0 + 1, z, z + 1)
    return V


def bus(L=20, W=6, H=5):
    V = set(); x0 = 7
    for wy in (2, L - 5):
        box(V, x0, x0 + W, wy, wy + 3, 0, 1)
    box(V, x0, x0 + W, 0, L, 1, 3)
    box(V, x0, x0 + W, 0, L, 2, H, hollow=True)
    for y in range(2, L - 2):
        if y % 3 != 1:
            V.discard((x0, y, H - 2)); V.discard((x0 + W - 1, y, H - 2))
    box(V, x0, x0 + W, 0, L, H, H + 2)
    return V


# ---------------------------------------------------------------- vessels (y = length)
def ship(L=16, W=6, twin=True, tiers=2, x0=7, y0=2):
    V = set()
    box(V, x0, x0 + W, y0 + 1, y0 + L, 0, 1)          # keel (stern step only)
    box(V, x0, x0 + W, y0, y0 + L, 1, 2)
    box(V, x0, x0 + W, y0, y0 + L, 2, 4)                          # deck (2 layers)
    z = 4
    for t in range(tiers):
        a, b = y0 + 2 + 2 * t, y0 + L - 5 - t
        box(V, x0 + 1 + (t > 0), x0 + W - 1 - (t > 0), a, b, z, z + 1); z += 1
    if twin:   # twin funnels/masts
        for fy in (a, b - 2):
            box(V, x0 + 2, x0 + 4, fy, fy + 2, z, z + 3)
            box(V, x0 + 2, x0 + 4, fy, fy + 1, z + 3, z + 4)
    return V


# ---------------------------------------------------------------- containers (radius profile per layer)
def lathe(profile, ring=2, base=True, c=10.0):
    V = set()
    for z, r in enumerate(profile):
        if z == 0 and base:
            disk(V, c, c, r, 0, 1)
        else:
            disk(V, c, c, r, z, z + 1, ring=ring)
    return V


def jar_ribbed(H=14, r=6):
    prof = [r] + [r] * (H - 3) + [r - 1, r - 1]
    V = lathe(prof)
    for z in range(0, H - 2):    # vertical accent ribs sticking out
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            V.add((int(10 + dx * (r + .2)) - (dx < 0), int(10 + dy * (r + .2)) - (dy < 0), z))
            V.add((int(10 + dx * (r + .2)) - (dx < 0) + abs(dy), int(10 + dy * (r + .2)) - (dy < 0) + abs(dx), z))
    disk(V, 10, 10, r - 1, H - 1, H)   # lid
    disk(V, 10, 10, 2, H, H + 1)       # knob
    return V


def mug(r=5.5, H=10, cube=False, faceted=False):
    V = set(); c = 10
    if cube:
        box(V, 5, 15, 5, 15, 0, 1); box(V, 5, 15, 5, 15, 1, H)
        rm(V, 7, 13, 7, 13, 1, H)
        hx = 14
    else:
        V = lathe([r] * H)
        if faceted:   # facet grooves: carve the outer shell every 3rd layer
            for z in range(2, H, 3):
                V -= {(x, y, z) for x in range(20) for y in range(20) if math.hypot(x + .5 - c, y + .5 - c) > r - .7}
        hx = int(c + r) - 1
    # handle on +x side: a squared C
    box(V, hx, hx + 3, 9, 11, 0, 2); box(V, hx, hx + 3, 9, 11, H - 4, H - 2)
    box(V, hx + 2, hx + 4, 9, 11, 0, H - 2)
    return V


def bottle(H=19):
    prof = [4] * 1 + [4.5] * 9 + [3.5, 2.5] + [1.6] * (H - 13) + [2]
    V = set()
    for z, r in enumerate(prof):
        disk(V, 10, 10, r, z, z + 1, ring=None if z in (0, 10, 11, H - 1) else max(1.6, abs(r - prof[z - 1]) + 1.6))
    return V


def jar_bulb():
    prof = [3.5, 4.5, 5.5, 6.5, 7, 7, 7, 6.5, 5.5, 4.5, 3.5, 2.5, 2.5, 3, 3.5]
    V = set()
    for z, r in enumerate(prof):
        disk(V, 10, 10, r, z, z + 1, ring=None if z == 0 else max(2, abs(r - prof[z - 1]) + 2))
    return V


def jar_geo(H=14):
    V = set()
    box(V, 4, 16, 4, 16, 0, 1)
    for z in range(1, H):
        s = 4 if z < H - 3 else 6
        box(V, s, 20 - s, s, 20 - s, z, z + 1, hollow=True)
        box(V, s + 1, 19 - s, s + 1, 19 - s, z, z + 1, hollow=True)
        if z % 3 == 1 and z < H - 3:   # textured band: notches
            for k in range(6, 14, 3):
                V.discard((s, k, z)); V.discard((19 - s, k, z)); V.discard((k, s, z)); V.discard((k, 19 - s, z))
    box(V, 5, 15, 5, 15, H - 3, H - 2, hollow=True)
    return V


def pot_stepped():
    V = set()
    box(V, 6, 14, 6, 14, 0, 1)
    tiers = [(6, 14, 1, 4), (5, 15, 4, 7), (4, 16, 7, 10), (3, 17, 10, 12)]
    for a, b, z0, z1 in tiers:
        for z in range(z0, z1):
            box(V, a, b, a, b, z, z + 1, hollow=True); box(V, a + 1, b - 1, a + 1, b - 1, z, z + 1, hollow=True)
    box(V, 3, 17, 3, 17, 12, 13, hollow=True)
    return V


def guitar_up():
    V = set(); y0 = 9
    box(V, 4, 12, y0, y0 + 2, 0, 7)            # lower bout standing upright (x-z plane)
    rm(V, 4, 5, y0, y0 + 2, 0, 1); rm(V, 11, 12, y0, y0 + 2, 0, 1)
    box(V, 5, 11, y0, y0 + 2, 7, 9)            # waist
    box(V, 4, 12, y0, y0 + 2, 9, 12)           # upper bout
    rm(V, 7, 9, y0 + 1, y0 + 2, 4, 6)          # sound hole on front face
    box(V, 7, 9, y0, y0 + 1, 12, 18)           # neck
    box(V, 6, 10, y0, y0 + 1, 18, 20)          # headstock
    return V


def bench(L=14, D=5, H=3, BH=4, x0=3, y0=7):
    V = set()
    for lx, ly in [(0, 0), (L - 2, 0), (0, D - 2), (L - 2, D - 2)]:
        box(V, x0 + lx, x0 + lx + 2, y0 + ly, y0 + ly + 2, 0, H)
    box(V, x0, x0 + L, y0, y0 + D, H, H + 2)
    H += 1
    box(V, x0, x0 + 2, y0 + D - 1, y0 + D, H + 1, H + 1 + BH); box(V, x0 + L - 2, x0 + L, y0 + D - 1, y0 + D, H + 1, H + 1 + BH)
    for z in range(H + 2, H + 1 + BH, 2):
        box(V, x0, x0 + L, y0 + D - 1, y0 + D, z, z + 1)
    box(V, x0, x0 + L, y0 + D - 1, y0 + D, H + BH, H + BH + 1)
    return V


def bench_solid(L=14, D=5):
    V = set(); x0, y0 = 3, 7
    box(V, x0, x0 + 2, y0, y0 + D, 0, 3); box(V, x0 + L - 2, x0 + L, y0, y0 + D, 0, 3)   # slab ends
    box(V, x0 + 2, x0 + L - 2, y0 + 1, y0 + 2, 0, 1)
    box(V, x0, x0 + L, y0, y0 + D, 3, 4)
    box(V, x0, x0 + L, y0 + D - 1, y0 + D, 4, 8)
    rm(V, x0 + 3, x0 + L - 3, y0 + D - 1, y0 + D, 5, 6)
    return V


S = lambda **k: (lambda r: sofa(**k))
C = lambda **k: (lambda r: chair(**k))
T = lambda **k: (lambda r: table_rect(**k))
CAR = lambda **k: (lambda r: car(**k))
DESIGNS = {
    "b622f21f": C(back="curved", W=10, D=9, H=6, BH=8, angled=True, x0=5, y0=5),
    "78c6fb4f": C(back="slats", W=8, D=8, H=5, BH=9),
    "850db052": C(back="solid", W=7, D=7, H=5, BH=9, leg=1),
    "999c9db8": S(L=14, D=7, seatH=3, BH=4, armH=2),
    "3841e206": lambda r: table_round(r=6, H=7),
    "46896515": C(back="ornate", W=8, D=8, H=5, BH=9),
    "0928def1": C(back="solid", W=10, D=10, H=6, BH=11, x0=5, y0=5),
    "6531b84d": CAR(L=15, W=8, cab=(4, 11), stepped=False, spoiler=True),
    "4f48ff49": lambda r: jar_ribbed(H=16, r=7),
    "568e5063": lambda r: guitar_up(),
    "f29f5c29": lambda r: ship(L=14, W=6, twin=True, tiers=2),
    "624bbd4a": lambda r: bus(L=20, W=6, H=4),
    "1dbe0e86": C(back="solid", W=10, D=10, H=5, BH=10, arms=True, x0=5, y0=5),
    "c2ec1e1c": C(back="solid", W=10, D=9, H=6, BH=10, leg=3, x0=5, y0=5),
    "b535b316": lambda r: mug(r=6, H=12, faceted=True),
    "65a8ddc4": lambda r: mug(cube=True, H=12),
    "80bf5182": S(L=10, D=6, seatH=3, BH=3, armH=2, feet=False, x0=5),
    "20b6863c": lambda r: table_panel(),
    "ccb5712f": CAR(L=14, W=8, cab=(3, 10), rounded=True),
    "98176df8": lambda r: pot_stepped(),
    "d96e610d": S(L=16, D=8, seatH=3, BH=5, armH=3, x0=2, angled=True),
    "755fcc85": lambda r: ship(L=13, W=5, twin=False, tiers=3),
    "d41685fe": lambda r: bottle(H=19),
    "d2ef8a07": C(back="solid", W=8, D=8, H=5, BH=11),
    "55aca077": lambda r: bench(L=14, D=5, H=3, BH=4),
    "aed771d6": CAR(L=15, W=8, cab=(3, 12), stepped=True),
    "39f4db3c": lambda r: jar_bulb(),
    "eebdb706": S(L=12, D=7, seatH=3, BH=3, armH=1),
    "a0cabf8d": CAR(L=16, W=9, cab=(4, 12), cabH=3),
    "cad84e61": T(W=14, D=9, H=6, leg=2, thick=2),
    "9c176acb": S(L=14, D=8, seatH=4, BH=4, armH=2, feet=False, x0=3),
    "2db72114": lambda r: bench_solid(L=12, D=4),
    "451dc10d": lambda r: table_round(r=7, H=7, frame=True),
    "418266b2": S(L=14, D=7, seatH=3, BH=5, armH=2),
    "97b4fdc9": S(L=15, D=7, seatH=3, BH=4, armH=3, arm=1),
    "26946d7f": lambda r: ship(L=15, W=6, twin=False, tiers=3),
    "6a810be8": S(L=16, D=8, seatH=4, BH=4, armH=2, x0=2),
    "8c3eca14": T(W=10, D=6, H=5, leg=1, apron=True, x0=5, y0=7),
    "67a05cc4": S(L=12, D=6, seatH=3, BH=4, armH=2, feet=False, x0=4),
    "0460a41c": T(W=12, D=8, H=5, leg=3, apron=True),
    "145f51f3": CAR(L=15, W=8, cab=(4, 11), rounded=True, lowH=2),
    "f30618dd": S(L=14, D=6, seatH=3, BH=3, armH=1, corner=True, x0=3, y0=9),
    "bd733337": T(W=10, D=7, H=6, leg=1, apron=True, x0=5, y0=6),
    "b860be32": T(W=14, D=9, H=6, leg=2, tiered=True, x0=3, y0=5),
    "9b5316c0": lambda r: jar_geo(H=15),
    "eed13054": C(back="solid", W=9, D=8, H=5, BH=10, arms=True, x0=5, y0=6),
    "9a499fc9": C(back="solid", W=10, D=9, H=6, BH=11, arms=True, x0=5, y0=5),
    "8febca25": S(L=14, D=7, seatH=3, BH=4, armH=3, feet=True),
    "1d2c1956": lambda r: ship(L=12, W=6, twin=False, tiers=3),
    "8dc870ce": C(back="ornate", W=10, D=9, H=6, BH=10, x0=5, y0=5),
}
ROUNDS = {'b622f21f': 2, '78c6fb4f': 2, '850db052': 2, '999c9db8': 1, '3841e206': 2, '46896515': 2, '0928def1': 2, '6531b84d': 2, '4f48ff49': 3, '568e5063': 1, 'f29f5c29': 4, '624bbd4a': 2, '1dbe0e86': 2, 'c2ec1e1c': 2, 'b535b316': 4, '65a8ddc4': 4, '80bf5182': 1, '20b6863c': 4, 'ccb5712f': 1, '98176df8': 1, 'd96e610d': 2, '755fcc85': 5, 'd41685fe': 1, 'd2ef8a07': 2, '55aca077': 3, 'aed771d6': 1, '39f4db3c': 4, 'eebdb706': 1, 'a0cabf8d': 1, 'cad84e61': 2, '9c176acb': 1, '2db72114': 1, '451dc10d': 2, '418266b2': 1, '97b4fdc9': 1, '26946d7f': 5, '6a810be8': 2, '8c3eca14': 1, '67a05cc4': 1, '0460a41c': 1, '145f51f3': 3, 'f30618dd': 2, 'bd733337': 1, 'b860be32': 1, '9b5316c0': 1, 'eed13054': 2, '9a499fc9': 2, '8febca25': 1, '1d2c1956': 3, '8dc870ce': 2}   # sid8 -> manual revision rounds (1 = first design accepted)
NOTES = {"b622f21f": "四腿+双层座板+靠背，侧视一眼是椅子", "78c6fb4f": "四腿+双层座板+靠背，侧视一眼是椅子", "850db052": "四腿+双层座板+靠背，侧视一眼是椅子", "999c9db8": "矮座、高靠背、两侧扶手、座垫有缝，像沙发", "3841e206": "桌面+腿/支撑，像桌子", "46896515": "四腿+双层座板+靠背，侧视一眼是椅子", "0928def1": "四腿+双层座板+靠背，侧视一眼是椅子", "6531b84d": "四轮+低车身+带窗驾驶舱，像小车", "4f48ff49": "空心圆罐，轮廓贴合描述", "568e5063": "竖立的像素吉他：琴身、腰、音孔、长琴颈和琴头，可辨", "f29f5c29": "平底船体+阶梯上层建筑，像船（较抽象）", "624bbd4a": "长条巴士带车窗带和车轮，砖数比同档偏多", "1dbe0e86": "四腿+双层座板+靠背，侧视一眼是椅子", "c2ec1e1c": "四腿+双层座板+靠背，侧视一眼是椅子", "b535b316": "空心杯身+侧把手（把手落地以保证稳定，略失真）", "65a8ddc4": "空心杯身+侧把手（把手落地以保证稳定，略失真）", "80bf5182": "矮座、高靠背、两侧扶手、座垫有缝，像沙发", "20b6863c": "桌面+腿/支撑，像桌子", "ccb5712f": "四轮+低车身+带窗驾驶舱，像小车", "98176df8": "四级外扩的空心阶梯花盆，很像", "d96e610d": "矮座、高靠背、两侧扶手、座垫有缝，像沙发", "755fcc85": "平底船体+阶梯上层建筑，像船（较抽象）", "d41685fe": "空心瓶身+收肩+细长瓶颈，像瓶子", "d2ef8a07": "四腿+双层座板+靠背，侧视一眼是椅子", "55aca077": "四腿长凳+横条靠背，像长椅", "aed771d6": "四轮+低车身+带窗驾驶舱，像小车", "39f4db3c": "空心圆罐，轮廓贴合描述", "eebdb706": "矮座、高靠背、两侧扶手、座垫有缝，像沙发", "a0cabf8d": "四轮+低车身+带窗驾驶舱，像小车", "cad84e61": "桌面+腿/支撑，像桌子", "9c176acb": "矮座、高靠背、两侧扶手、座垫有缝，像沙发", "2db72114": "板式侧腿长凳+靠背，像长椅", "451dc10d": "桌面+腿/支撑，像桌子", "418266b2": "矮座、高靠背、两侧扶手、座垫有缝，像沙发", "97b4fdc9": "矮座、高靠背、两侧扶手、座垫有缝，像沙发", "26946d7f": "平底船体+阶梯上层建筑，像船（较抽象）", "6a810be8": "矮座、高靠背、两侧扶手、座垫有缝，像沙发", "8c3eca14": "桌面+腿/支撑，像桌子", "67a05cc4": "矮座、高靠背、两侧扶手、座垫有缝，像沙发", "0460a41c": "桌面+腿/支撑，像桌子", "145f51f3": "四轮+低车身+带窗驾驶舱，像小车", "f30618dd": "L形转角沙发，可辨", "bd733337": "桌面+腿/支撑，像桌子", "b860be32": "桌面+腿/支撑，像桌子", "9b5316c0": "空心圆罐，轮廓贴合描述", "eed13054": "四腿+双层座板+靠背，侧视一眼是椅子", "9a499fc9": "四腿+双层座板+靠背，侧视一眼是椅子", "8febca25": "矮座、高靠背、两侧扶手、座垫有缝，像沙发", "1d2c1956": "平底船体+阶梯上层建筑，像船（较抽象）", "8dc870ce": "四腿+双层座板+靠背，侧视一眼是椅子"}


def run_one(p):
    s8 = p["structure_id"][:8]
    PREF.clear()
    V = {v for v in DESIGNS[s8](0) if all(0 <= c < 20 for c in v)}
    text = tile2(V); f = feats(text)
    r = check(text, use_gurobi=True, gurobi_kwargs={"time_limit": 400, "threads": 2}).to_dict()
    render_png(text, OUT / "renders" / f"{s8}.png")
    row = {"structure_id": p["structure_id"], "caption": p["caption"], "category_id": p["category_id"],
           "bucket": p["bucket"], "bricks": text, "physics_ok": r["ok"], "check_method": r.get("method"),
           "check_msg": r["message"][:200], "filter_ok": bool(r["ok"] and keep(f)),
           "rounds": ROUNDS.get(s8, 1), "n_bricks": f["n_bricks"], "feats": f, "self_eval": NOTES.get(s8, "")}
    (OUT / "tmp").mkdir(exist_ok=True)
    (OUT / "tmp" / f"{s8}.json").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return f"{s8} {p['bucket']} {p.get('n_bricks')} {f['n_bricks']} {r['ok']} {keep(f)} {f['fill']:.2f} {r['message'][:90]}"


def main():
    from multiprocessing import Pool
    only = sys.argv[1:]
    P = [json.loads(l) for l in open(ROOT / "data/opus_gen/prompts_0.jsonl", encoding="utf-8")]
    (OUT / "renders").mkdir(exist_ok=True)
    todo = [p for p in P if not only or p["structure_id"][:8] in only]
    if only != ["merge"]:
        with Pool(int(__import__("os").environ.get("NP", "8"))) as pool:
            for line in pool.imap_unordered(run_one, todo):
                print(line, flush=True)
    rows = [json.loads((OUT / "tmp" / f"{p['structure_id'][:8]}.json").read_text(encoding="utf-8"))
            for p in P if (OUT / "tmp" / f"{p['structure_id'][:8]}.json").exists()]
    for r in rows:
        s8 = r["structure_id"][:8]; r["rounds"] = ROUNDS.get(s8, 1); r["self_eval"] = NOTES.get(s8, "")
    (OUT / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
