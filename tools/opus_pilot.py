"""Opus hand-designed pilot: 10 of the 20 harness_vision train prompts, shapes designed from caption only
(target bricks are used ONLY for the aligned-IoU metric, never for construction).
Each shape = set of voxels; each z-layer is tiled greedily with allowed bricks (orientation preference alternates
by layer so joints stagger). python tools/opus_pilot.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from brickexp.checker import check, gurobi_available, parse  # noqa: E402
from distill_filter import feats, keep  # noqa: E402
from distill_iou import vox  # noqa: E402
from distill_shape import aligned_iou  # noqa: E402


def render_png(text, path):
    """Same look as harness_vision.render_png but with fixed 0..20 axes (true aspect; the original autoscale
    stretches flat models into cubes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    b = parse(text)[0]
    v = np.zeros((20, 20, 20), bool); c = np.empty((20, 20, 20, 4)); cm = plt.get_cmap("tab20")
    for k, x in enumerate(b):
        v[x.x:x.x + x.h, x.y:x.y + x.w, x.z] = True; c[x.x:x.x + x.h, x.y:x.y + x.w, x.z] = cm(k % 20)
    fig = plt.figure(figsize=(6, 3))
    for i, (el, az) in enumerate(((25, -60), (25, 120))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.voxels(v, facecolors=c, edgecolor=(0, 0, 0, .25), linewidth=.3)
        ax.set_xlim(0, 20); ax.set_ylim(0, 20); ax.set_zlim(0, 20)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(el, az); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

OUT = ROOT / "results" / "opus_pilot"
SIZES = [(2, 6), (6, 2), (2, 4), (4, 2), (1, 8), (8, 1), (1, 6), (6, 1), (2, 2), (1, 4), (4, 1), (1, 2), (2, 1), (1, 1)]


def box(V, x0, x1, y0, y1, z0, z1, hollow=False):
    for x in range(x0, x1):
        for y in range(y0, y1):
            for z in range(z0, z1):
                if not hollow or x in (x0, x1 - 1) or y in (y0, y1 - 1):
                    V.add((x, y, z))


PREF = {}


def tile(V):
    lines = []
    for z in sorted({v[2] for v in V}):
        free = {(x, y) for x, y, zz in V if zz == z}
        order = PREF.get(z) or (SIZES if z % 2 == 0 else [(w, h) for h, w in SIZES])
        for x, y in sorted(free, key=lambda p: (p, p[::-1])[z % 2]):
            if (x, y) not in free:
                continue
            for h, w in order:
                cells = [(x + i, y + j) for i in range(h) for j in range(w)]
                if all(c in free for c in cells):
                    free -= set(cells); lines.append(f"{h}x{w} ({x},{y},{z})"); break
    return "\n".join(lines)


# ---------------- designs (caption -> voxels); version argument = revision round ----------------
def guitar(r):
    V = set()
    box(V, 1, 9, 2, 9, 0, 2)            # lower bout (body), lying flat
    box(V, 2, 8, 1, 10, 0, 1)
    box(V, 9, 11, 3, 8, 0, 2)           # waist
    box(V, 11, 16, 3, 8, 0, 2)          # upper bout
    for v in [(4, 5, 1), (5, 5, 1), (4, 6, 1), (5, 6, 1)]:
        V.discard(v)                    # sound hole
    box(V, 16, 19, 5, 7, 0, 1)          # neck (long, 2 wide)
    box(V, 16, 19, 5, 7, 1, 2) if r >= 1 else None
    box(V, 18, 20, 4, 8, 0, 1)          # headstock
    return V


def guitar_v(r):  # elongated: neck should be long relative to body -> body x0..9, neck x9..18, head 18..20
    V = set()
    box(V, 0, 8, 1, 9, 0, 2); box(V, 1, 7, 0, 10, 0, 1)   # body
    for v in [(3, 4, 1), (4, 4, 1), (3, 5, 1), (4, 5, 1)]:
        V.discard(v)
    box(V, 8, 18, 4, 6, 0, 2)          # neck
    box(V, 18, 20, 3, 7, 0, 2)         # headstock
    return V


def table(r):
    V = set()
    for x, y in [(0, 0), (8, 0), (0, 6), (8, 6)]:
        box(V, x, x + 2, y, y + 2, 0, 4)      # blocky 2x2 legs
    box(V, 0, 10, 0, 8, 4, 6)                 # thick top
    box(V, 0, 10, 0, 8, 3, 4, hollow=True)    # apron ring
    return V


def bench(r):
    V = set()
    for x, y in [(0, 0), (0, 14), (4, 0), (4, 14)]:
        box(V, x, x + 2, y, y + 2, 0, 3)      # legs (x=depth, y=length 16)
    box(V, 0, 6, 0, 16, 2, 3, hollow=True)    # apron frame under seat
    for y in range(0, 16, 2):
        box(V, 0, 6, y, y + 1, 3, 4)          # slats across the seat, gaps between
    box(V, 0, 6, 15, 16, 3, 4)
    for z in (4, 6, 8):
        box(V, 4, 6, 0, 2, z, z + 1); box(V, 4, 6, 14, 16, z, z + 1)  # back posts (between slats)
    for z in (5, 7, 9):
        box(V, 4, 6, 0, 16, z, z + 1)         # back slats, each 1x8 piece lands on a post
        PREF[z] = [(1, 8), (1, 1)]
    return V


def train(r):
    V = set()
    box(V, 1, 7, 0, 18, 0, 1)                 # chassis
    for y in (1, 6, 11, 15):
        box(V, 0, 8, y, y + 2, 0, 1)          # wheels poke out sides
    box(V, 1, 7, 0, 18, 1, 5, hollow=True)    # body walls
    for y in (3, 4, 8, 9, 13, 14):
        V.discard((1, y, 3)); V.discard((6, y, 3))   # windows
    box(V, 0, 8, 0, 18, 5, 6)                 # roof tier 1 (eaves)
    box(V, 2, 6, 1, 17, 6, 7)                 # tier 2
    box(V, 3, 5, 3, 15, 7, 8)                 # tier 3 (clerestory)
    return V


def car(r):
    V = set()
    for x, y in [(0, 0), (6, 0), (0, 12), (6, 12)]:
        box(V, x, x + 2, y, y + 3, 0, 1)      # wheels
    box(V, 0, 8, 0, 15, 1, 3)                 # boxy lower body
    box(V, 1, 7, 3, 11, 3, 5, hollow=True)    # cabin
    box(V, 1, 7, 3, 11, 5, 6)                 # roof
    box(V, 2, 6, 14, 15, 3, 4)                # hood block / bumper detail
    return V


def bottle(r):
    V = set()
    box(V, 5, 15, 5, 15, 0, 1)                # square base plate
    box(V, 5, 15, 5, 15, 1, 9, hollow=True)   # square body
    box(V, 5, 15, 5, 15, 9, 10)               # shoulder lid
    box(V, 7, 13, 7, 13, 10, 11)              # shoulder step
    box(V, 8, 12, 8, 12, 11, 18, hollow=True) # tall neck
    box(V, 8, 12, 8, 12, 18, 19)              # cap
    return V


def bus(r):
    V = set()
    box(V, 0, 8, 0, 20, 0, 1)                 # chassis
    box(V, 0, 8, 0, 20, 1, 6, hollow=True)
    for y in range(2, 18):
        if y % 3 != 1:
            for z in (3, 4):
                V.discard((0, y, z)); V.discard((7, y, z))   # window band
    box(V, 0, 8, 0, 20, 6, 7)                 # roof: tiled with 8-long bricks spanning both walls
    PREF[6] = [(8, 1), (8, 2), (4, 1), (2, 1), (1, 1)]
    return V


def pot(r):
    V = set()
    box(V, 5, 13, 5, 13, 0, 1)
    box(V, 5, 13, 5, 13, 1, 4, hollow=True)   # tier 1 base
    box(V, 4, 14, 4, 14, 4, 5)                # ledge
    box(V, 4, 14, 4, 14, 5, 9, hollow=True)   # tier 2
    box(V, 3, 15, 3, 15, 9, 10)               # ledge
    box(V, 3, 15, 3, 15, 10, 14, hollow=True) # tier 3
    box(V, 2, 16, 2, 16, 14, 15)              # rim: 2-thick ring overlapping the wall below
    box(V, 4, 14, 4, 14, 14, 15)
    V -= {(x, y, 14) for x in range(4, 14) for y in range(4, 14)}
    return V


def chair(r):
    V = set()
    for x, y in [(0, 0), (6, 0), (0, 6), (6, 6)]:
        box(V, x, x + 2, y, y + 2, 0, 4)
    box(V, 0, 8, 0, 8, 4, 5)                  # seat
    box(V, 0, 8, 0, 8, 5, 6)
    for k, z in enumerate(range(6, 13)):      # angled backrest: leans back (+y) one step every 2 layers
        y0 = 6 + k // 2
        box(V, 0, 8, y0, y0 + 2, z, z + 1)
    return V


def vessel(r):
    V = set()
    box(V, 6, 10, 3, 17, 0, 1)                # keel
    box(V, 5, 11, 2, 18, 1, 2)                # lower hull
    box(V, 4, 12, 1, 19, 2, 3)                # hull flaring (2-thick ring)
    V -= {(x, y, 2) for x in range(6, 10) for y in range(3, 17)}
    box(V, 4, 12, 1, 19, 3, 4)                # deck
    box(V, 5, 11, 4, 15, 4, 6, hollow=True)   # tier 1 cabin
    box(V, 5, 11, 4, 15, 6, 7)
    box(V, 6, 10, 6, 13, 7, 8)                # tier 2
    box(V, 7, 9, 8, 11, 8, 10)                # tier 3 (bridge/funnel)
    return V


DESIGNS = {"c59ff7bc": guitar_v, "88ae8881": table, "bc7d1523": bench, "3f6b3d88": train, "ea150399": car,
           "9252da4b": bottle, "0e0664ab": bus, "9751a363": pot, "4c39a999": chair, "b9f8fe18": vessel}


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "renders").mkdir(exist_ok=True)
    eps = {}
    for l in open(ROOT / "results/harness_vision/episodes.jsonl", encoding="utf-8"):
        e = json.loads(l); eps.setdefault(e["structure_id"], {})[e["arm"]] = e
    from brickexp.data import load_split
    tgt = {r["structure_id"]: r for r in load_split("train") if r["structure_id"][:8] in DESIGNS}
    only = sys.argv[1:]
    rows = []
    for sid8, fn in DESIGNS.items():
        if only and sid8 not in only:
            continue
        rec = next(r for k, r in tgt.items() if k.startswith(sid8))
        PREF.clear(); t0 = time.time(); hist = []
        for rnd in range(3):
            text = tile(fn(rnd)); r = check(text, use_gurobi=True).to_dict(); f = feats(text)
            hist.append({"round": rnd, "ok": r["ok"], "msg": r["message"], "method": r.get("method"), "keep": keep(f)})
            if r["ok"] and keep(f):
                break
        render_png(text, OUT / "renders" / f"{sid8}.png")
        row = {"structure_id": rec["structure_id"], "bucket": rec["bucket"], "caption": rec["caption"],
               "pass": r["ok"], "method": r.get("method"), "diversity_ok": bool(r["ok"] and keep(f)), **f,
               "iou_aligned": aligned_iou(vox(text), vox(rec["bricks"])), "rounds": len(hist), "hist": hist,
               "sec": round(time.time() - t0, 1),
               "iou_A": eps[rec["structure_id"]]["A"].get("iou_aligned"), "iou_B": eps[rec["structure_id"]]["B"].get("iou_aligned"),
               "text": text}
        rows.append(row)
        print(sid8, row["pass"], row["diversity_ok"], round(row["iou_aligned"], 3), row["n_bricks"], hist[-1]["msg"][:120])
    if not only:
        (OUT / "samples.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
        print("gurobi", gurobi_available())


if __name__ == "__main__":
    main()
