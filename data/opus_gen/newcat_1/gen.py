"""newcat_1 generator: tree / lamp (out-of-dataset categories), caption-only parametric designs.
Reuses batch_0/gen.py (support-aware tile2, shape-function style) + opus_pilot box/render_png;
checked with brickexp.checker.check(use_gurobi=True), filtered with distill_filter.keep.
python data/opus_gen/newcat_1/gen.py [id ...]      (NP env = processes; tmp rows go to NC1_TMP)
"""
import json, math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "data/opus_gen/batch_0"))
import types  # noqa: E402
try:
    import scipy  # noqa: F401
except ImportError:
    sys.modules["distill_shape"] = types.SimpleNamespace(aligned_iou=None)
import opus_pilot as OP  # noqa: E402
from opus_pilot import box, render_png  # noqa: E402
import importlib.util as _iu  # noqa: E402
_sp = _iu.spec_from_file_location("batch0_gen", ROOT / "data/opus_gen/batch_0/gen.py"); _B = _iu.module_from_spec(_sp); _sp.loader.exec_module(_B)
tile2, disk, rm = _B.tile2, _B.disk, _B.rm   # batch_0 support-aware tiler + helpers
from brickexp.checker import check  # noqa: E402
from distill_filter import feats, keep  # noqa: E402

OUT = Path(__file__).resolve().parent
TMP = Path(os.environ.get("NC1_TMP", str(OUT / "nc1_tmp")))
C = 10   # grid centre


def ring(V, r, z0, z1, th=1, cx=C, cy=C):
    disk(V, cx, cy, r, z0, z1, ring=th)


def corbel(V, r0, r1, z, cx=C, cy=C, ring_th=None):
    """grow a disk from radius r0 to r1 by <=1.5 per layer starting at z (supports an overhang); returns next z."""
    r = r0
    while r < r1:
        r = min(r1, r + 1.0)
        disk(V, cx, cy, r, z, z + 1, ring=ring_th if r >= r1 else None); z += 1
    return z


def sq_corbel(V, h0, h1, z, cx=C, cy=C):
    """square version: half-width h0 -> h1, +1 per layer."""
    h = h0
    while h < h1:
        h += 1; box(V, cx - h, cx + h, cy - h, cy + h, z, z + 1); z += 1
    return z


def trunk(V, z0, z1, w=2, cx=C, cy=C):
    box(V, cx - w // 2, cx - w // 2 + w, cy - w // 2, cy - w // 2 + w, z0, z1)


def root(V, w=4):   # flared foot so a thin trunk is not a 1-brick pin
    box(V, C - w // 2, C + w // 2, C - w // 2, C + w // 2, 0, 1)


def blob(V, cx, cy, zc, R, Rz=None, shell=None):
    Rz = Rz or R
    for z in range(max(0, int(zc - Rz)), min(20, int(zc + Rz) + 1)):
        t = 1 - ((z + .5 - zc) / Rz) ** 2
        if t > 0:
            disk(V, cx, cy, R * math.sqrt(t), z, z + 1, ring=shell)


# ------------------------------------------------------------------ trees
def tree_round(H=9, R=6, Rz=5, tw=2, shell=2):
    V = set(); root(V); trunk(V, 0, H + 2, tw)
    z = corbel(V, 1, R - 1.5, H - 2)
    blob(V, C, C, H + Rz - 1, R, Rz, shell)
    disk(V, C, C, R - 1, min(19, int(H + 2 * Rz - 2)), min(20, int(H + 2 * Rz - 1)))
    return V


def tree_cone(tiers=((3, 6, 4), (7, 5, 4), (11, 4, 4), (15, 2, 3)), top=19, tw=2, shell=None):
    V = set(); root(V); trunk(V, 0, top, tw)
    for z0, r, h in tiers:   # each tier: corbel out then taper
        zz = corbel(V, 1.5, r, z0)
        for k in range(h - (zz - z0)):
            disk(V, C, C, max(1.5, r - k * 1.2), zz + k, zz + k + 1, ring=shell if r - k * 1.2 > 3 else None)
    disk(V, C, C, 1, top - 1, top)
    return V


def cypress():
    V = set(); root(V, 4); trunk(V, 0, 4, 2)
    blob(V, C, C, 11, 3.6, 8.5, shell=None)
    rm(V, C - 1, C + 1, C - 1, C + 1, 6, 15)   # hollow core keeps fill down
    return V


def palm():
    V = set(); root(V, 4)
    for z in range(1, 16):   # slender, slightly leaning trunk
        o = 0; box(V, C - 1 + o, C + 1 + o, C - 1, C + 1, z, z + 1)
    cx = C
    disk(V, cx, C, 2.5, 15, 18)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
        for k in range(1, 6):
            x, y = int(cx + dx * k), int(C + dy * k)
            if 0 <= x < 20 and 0 <= y < 20:
                z = 16 if k < 4 else 15        # fronds droop toward tips
                box(V, x - 1, x + 1, y - 1, y + 1, z, z + 1)
                if k < 4:
                    box(V, x - 1, x + 1, y - 1, y + 1, 17, 18)
    return V


def mushroom_tree(H=8, R=8):
    V = set(); box(V, C - 3, C + 3, C - 3, C + 3, 0, 1); trunk(V, 1, H + 1, 3)
    z = corbel(V, 2, R, H)
    for k in range(5):
        disk(V, C, C, R - k * 1.6, z + k, z + k + 1, ring=3 if R - k * 1.6 > 5 else None)
    return V


def block_tree(H=8, h=7, Hc=8):
    V = set(); root(V, 4); trunk(V, 0, H + 2, 2)
    z = sq_corbel(V, 1, h, H - 3)
    box(V, C - h, C + h, C - h, C + h, z, z + Hc, hollow=True)
    box(V, C - h, C + h, C - h, C + h, z + Hc - 2, z + Hc)
    box(V, C - h, C + h, C - h, C + h, z, z + 1)
    return V


def oak(H=6, R=9.5):
    V = set(); box(V, C - 2, C + 2, C - 2, C + 2, 0, 1); trunk(V, 0, H + 1, 4)
    z = corbel(V, 2, R, H - 1)
    disk(V, C, C, R, z, z + 1); disk(V, C, C, R - 1.5, z + 1, z + 2); disk(V, C, C, R - 4, z + 2, z + 3)
    return V


def bonsai():
    V = set(); box(V, C - 4, C + 4, C - 3, C + 3, 0, 2); rm(V, C - 2, C + 2, C - 1, C + 1, 0, 1)
    trunk(V, 2, 5, 2); box(V, C, C + 2, C - 1, C + 1, 4, 6)
    disk(V, C + 1, C, 5, 6, 7); disk(V, C + 1, C, 3.5, 7, 8)
    return V


def sapling():
    V = set(); box(V, C - 2, C + 2, C - 2, C + 2, 0, 1); trunk(V, 1, 18, 2)
    for z, dx, dy in ((7, 1, 0), (11, -1, 0), (14, 0, 1), (17, 0, -1)):   # a few small leaf twigs
        x, y = C + dx * 2, C + dy * 2
        box(V, min(C, x) - 1 if dx < 0 else C - 1, max(C + 1, x + 2), min(C, y) - 1 if dy < 0 else C - 1, max(C + 1, y + 2), z, z + 1)
        box(V, x - 1 + dx, x + 1 + dx, y - 1 + dy, y + 1 + dy, z, z + 2)
    disk(V, C, C, 2, 18, 20)
    return V


def two_branch():
    V = set(); box(V, C - 2, C + 2, C - 2, C + 2, 0, 1); trunk(V, 1, 9, 4)
    for s in (-1, 1):
        for k in range(6):   # diagonal staircase branch
            x = C - 1 + s * (k + 1)
            box(V, x - 1, x + 3, C - 2, C + 2, 7 + k, 8 + k)
        blob(V, C + s * 6, C, 15.5, 3.2, 3.2, shell=None)
    return V


def planter_tree():
    V = set(); box(V, C - 4, C + 4, C - 4, C + 4, 0, 4); rm(V, C - 2, C + 2, C - 2, C + 2, 0, 2)
    trunk(V, 4, 10, 2)
    z = corbel(V, 1, 5, 8)
    blob(V, C, C, 14, 5.5, 4.5, shell=2)
    return V


def three_layer():
    V = set(); root(V, 4); trunk(V, 0, 20, 2)
    for z0, r in ((4, 7), (9, 6), (14, 5)):
        z = corbel(V, 1, r, z0)
        disk(V, C, C, r, z, z + 1); disk(V, C, C, r - 2, z + 1, z + 2)
    return V


def willow():
    V = set(); root(V, 4); trunk(V, 0, 12, 2)
    z = corbel(V, 1, 7, 10)
    disk(V, C, C, 7, z, z + 1); disk(V, C, C, 5, z + 1, z + 2); disk(V, C, C, 3, z + 2, z + 3)
    ring(V, 7, z - 4, z, 1)   # drooping curtain hanging from the rim
    return V


def bushy(H=4, R=8):
    V = set(); box(V, C - 3, C + 3, C - 3, C + 3, 0, 1); trunk(V, 0, H + 3, 4)
    corbel(V, 2, R - 1, H)
    blob(V, C, C, H + 7, R, 7, shell=2)
    return V


# ------------------------------------------------------------------ lamps
def base_round(r=5, h=1):
    V = set(); disk(V, C, C, r, 0, h); return V


def shade_cone(V, z0, rb, rt, h, ring_th=1, cap=True):
    """lamp shade widening downward: built top-down as a corbelled solid top then hollow walls."""
    for k in range(h):
        r = rb - (rb - rt) * k / max(1, h - 1)
        disk(V, C, C, r, z0 + k, z0 + k + 1, ring=None if (cap and k == h - 1) else ring_th + 1.5)


def table_lamp():
    V = base_round(4, 1); disk(V, C, C, 3, 1, 2); trunk(V, 2, 9, 2)
    corbel(V, 1, 3, 8)
    shade_cone(V, 10, 5, 3, 5, cap=True)
    return V


def cube_lamp():
    V = set(); box(V, C - 3, C + 3, C - 3, C + 3, 0, 1); trunk(V, 1, 6, 2)
    sq_corbel(V, 1, 3, 5)
    box(V, C - 3, C + 3, C - 3, C + 3, 7, 11, hollow=True); box(V, C - 3, C + 3, C - 3, C + 3, 11, 12)
    return V


def twin_lamp():
    V = set(); box(V, C - 4, C + 4, C - 3, C + 3, 0, 2); box(V, C - 2, C + 2, C - 2, C + 2, 2, 3)
    trunk(V, 3, 10, 2)
    box(V, C - 7, C + 7, C - 1, C + 1, 10, 11); box(V, C - 6, C + 6, C - 2, C + 2, 9, 10)
    box(V, C - 7, C + 7, C - 1, C + 1, 11, 12)
    for sx in (C - 6, C + 6):
        disk(V, sx, C, 1, 12, 13)
        for k, r in enumerate((3.2, 3.2, 2.8, 2.5, 2.2)):
            disk(V, sx, C, r, 13 + k, 14 + k, ring=None if k in (0, 4) else 1)
    return V


def stepped_lamp():
    V = set()
    for k, h in enumerate((7, 6, 5, 4)):
        box(V, C - h, C + h, C - h, C + h, k, k + 1); rm(V, C - h + 2, C + h - 2, C - h + 2, C + h - 2, k, k + 1) if k < 2 else None
    for k, h in enumerate((7, 6, 5)):
        box(V, C - h + 1, C + h - 1, C - h + 1, C + h - 1, k + 1, k + 2) if False else None
    trunk(V, 4, 12, 2)
    corbel(V, 1, 3, 11)
    shade_cone(V, 13, 6, 3.5, 6)
    return V


def square_neck():
    V = set(); box(V, C - 4, C + 4, C - 4, C + 4, 0, 3); rm(V, C - 2, C + 2, C - 2, C + 2, 1, 2)
    trunk(V, 3, 13, 2)
    corbel(V, 1, 3, 12)
    shade_cone(V, 14, 5, 3, 5)
    return V


def cyl_body():
    V = set(); disk(V, C, C, 4, 0, 1); ring(V, 4, 1, 8, 1); disk(V, C, C, 4, 8, 9)
    trunk(V, 9, 11, 2)
    shade_cone(V, 11, 4.5, 3, 4)
    return V


def wide_cone():
    V = set(); disk(V, C, C, 5, 0, 1); disk(V, C, C, 3, 1, 2); trunk(V, 2, 6, 2)
    z = corbel(V, 1, 4, 5)
    for k in range(5):
        disk(V, C, C, 4 + k, z + k, z + k + 1, ring=None if k == 0 else 2)
    return V


def street_lamp():
    V = set(); box(V, C - 3, C + 3, C - 3, C + 3, 0, 2); trunk(V, 2, 18, 2)
    box(V, C - 1, C + 6, C - 1, C + 1, 17, 18); box(V, C - 1, C + 7, C - 1, C + 1, 18, 19)
    box(V, C + 3, C + 7, C - 2, C + 2, 16, 17)    # light head under the arm end
    box(V, C + 4, C + 6, C - 1, C + 1, 15, 16)
    box(V, C - 2, C + 2, C - 2, C + 2, 19, 20)
    return V


def lantern():
    V = set(); box(V, C - 3, C + 3, C - 3, C + 3, 0, 2)
    for x, y in ((C - 3, C - 3), (C + 2, C - 3), (C - 3, C + 2), (C + 2, C + 2)):
        box(V, x, x + 1, y, y + 1, 2, 8)
    disk(V, C, C, 1, 2, 7)
    box(V, C - 3, C + 3, C - 3, C + 3, 8, 9); box(V, C - 2, C + 2, C - 2, C + 2, 9, 10); box(V, C - 1, C + 1, C - 1, C + 1, 10, 12)
    return V


def mushroom_lamp():
    V = set(); disk(V, C, C, 4, 0, 1); trunk(V, 1, 6, 2)
    z = corbel(V, 1, 5.5, 5)
    blob(V, C, C, z - 1, 5.5, 4, shell=None)
    rm(V, 0, 20, 0, 20, 0, z - 1) if False else None
    V -= {(x, y, zz) for x, y, zz in V if zz < z and zz >= 5 and math.hypot(x + .5 - C, y + .5 - C) > 5.6}
    return V


def floor_lamp():
    V = set(); disk(V, C, C, 4, 0, 1); disk(V, C, C, 2, 1, 2); trunk(V, 2, 14, 2)
    corbel(V, 1, 3, 13)
    shade_cone(V, 15, 5, 3, 5)
    return V


def desk_lamp():
    V = set(); box(V, C - 4, C + 5, C - 4, C + 4, 0, 2); box(V, C - 2, C + 2, C - 2, C + 2, 2, 3)
    for k in range(9):   # arm angled up and over (staircase, 2 wide)
        x = C - 1 + k // 2 * 1 - 3
        box(V, x, x + 3, C - 1, C + 1, 3 + k, 4 + k)
    for k in range(4):   # forearm angled down toward the head
        x = C + 1 + k
        box(V, x, x + 3, C - 1, C + 1, 12 - k, 13 - k)
    box(V, C + 3, C + 8, C - 2, C + 2, 8, 9); box(V, C + 4, C + 7, C - 2, C + 2, 7, 8)   # small head
    box(V, C - 4, C + 5, C - 4, C + 4, 2, 3, hollow=True)
    return V


def arc_lamp():
    V = set(); box(V, C - 7, C - 1, C - 3, C + 3, 0, 2); box(V, C - 6, C - 2, C - 2, C + 2, 2, 3)
    trunk(V, 3, 15, 2, cx=C - 4)
    path = [(-4, 15), (-3, 16), (-2, 17), (0, 17), (2, 17), (3, 16), (4, 15)]
    for x, z in path:
        box(V, C + x - 1, C + x + 2, C - 1, C + 1, z, z + 1)
    box(V, C + 5, C + 6, C - 1, C + 1, 14, 15); box(V, C + 4, C + 7, C - 1, C + 1, 15, 16)
    box(V, C + 2, C + 9, C - 2, C + 2, 13, 14); box(V, C + 3, C + 8, C - 1, C + 1, 12, 13)   # hanging shade
    return V


def torch_lamp():
    V = set(); disk(V, C, C, 4, 0, 1); disk(V, C, C, 3, 1, 2); trunk(V, 2, 11, 2)
    z = corbel(V, 1, 7, 11)
    ring(V, 7, z, z + 3, 2)
    return V


def pendant():
    V = set(); disk(V, C, C, 9, 0, 1, ring=2)   # rim of a wide bell shade resting on its lip
    for k in range(1, 8):
        r = 9 - k * 0.9
        disk(V, C, C, r, k, k + 1, ring=2 if k < 7 else None)
    disk(V, C, C, 3, 8, 9); trunk(V, 9, 19, 2); box(V, C - 2, C + 2, C - 2, C + 2, 19, 20)   # cord + ceiling cup
    return V


DESIGNS = {
    "tree_05": sapling, "tree_04": two_branch, "tree_12": bonsai, "tree_13": planter_tree,
    "tree_01": lambda: tree_cone(), "tree_10": cypress,
    "tree_07": lambda: tree_cone(((2, 8, 4), (6, 6.5, 4), (10, 5, 4), (14, 3.5, 3)), top=19),
    "tree_02": bushy, "tree_08": block_tree, "tree_11": mushroom_tree, "tree_03": palm,
    "tree_00": lambda: tree_round(H=9, R=5, Rz=4.5), "tree_06": oak, "tree_14": three_layer, "tree_09": willow,
    "lamp_12": cube_lamp, "lamp_14": twin_lamp, "lamp_11": stepped_lamp, "lamp_07": square_neck,
    "lamp_06": cyl_body, "lamp_04": wide_cone, "lamp_00": table_lamp, "lamp_05": street_lamp,
    "lamp_03": lantern, "lamp_09": mushroom_lamp, "lamp_01": floor_lamp, "lamp_02": desk_lamp,
    "lamp_10": arc_lamp, "lamp_13": torch_lamp, "lamp_08": pendant,
}
ROUNDS = {k: 2 for k in ("lamp_11", "tree_12", "lamp_04", "tree_13", "tree_04", "tree_03", "tree_11")}
NOTES = {}  # free-text self-evaluation notes (originally in Chinese) removed in the public release; not used downstream

def prune(V):
    """tile, then drop bricks not connected (via vertical stud overlap) to the ground; retile until stable."""
    while True:
        text = tile2(V); B = [tuple(map(int, __import__("re").findall(r"\d+", l))) for l in text.splitlines()]
        cells = {}
        for k, (h, w, x, y, z) in enumerate(B):
            for i in range(h):
                for j in range(w):
                    cells[(x + i, y + j, z)] = k
        adj = {k: set() for k in range(len(B))}
        for (x, y, z), k in cells.items():
            o = cells.get((x, y, z + 1))
            if o is not None:
                adj[k].add(o); adj[o].add(k)
        seen = [k for k, b in enumerate(B) if b[4] == 0]; seen = set(seen); st = list(seen)
        while st:
            for o in adj[st.pop()]:
                if o not in seen:
                    seen.add(o); st.append(o)
        bad = {c for c, k in cells.items() if k not in seen}
        if not bad:
            return text
        V -= bad


def run_one(p):
    i = p["id"]
    OP.PREF.clear()
    V = {v for v in DESIGNS[i]() if all(0 <= c < 20 for c in v)}
    text = prune(V); f = feats(text)
    r = check(text, use_gurobi=True, gurobi_kwargs={"time_limit": 400, "threads": 2}).to_dict()
    render_png(text, OUT / "renders" / f"{i}.png")
    row = {"id": i, "newcat": p["newcat"], "caption": p["caption"], "bucket": p["bucket"], "bricks": text,
           "physics_ok": r["ok"], "check_msg": r["message"][:200], "filter_ok": bool(r["ok"] and keep(f)),
           "rounds": ROUNDS.get(i, 1), "n_bricks": f["n_bricks"], "feats": f, "self_eval": NOTES.get(i, "")}
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / f"nc1_{i}.json").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return f"{i} {p['bucket']} n={f['n_bricks']} phys={r['ok']} keep={keep(f)} fill={f['fill']:.2f} types={f['n_types']} share={f['max_type_share']:.2f} {r['message'][:80]}"


def main():
    from multiprocessing import Pool
    only = sys.argv[1:]
    P = [json.loads(l) for l in open(ROOT / "data/opus_gen/newcat_prompts_1.jsonl", encoding="utf-8")]
    (OUT / "renders").mkdir(exist_ok=True)
    todo = [p for p in P if not only or p["id"] in only]
    if only != ["merge"]:
        with Pool(int(os.environ.get("NP", "8"))) as pool:
            for line in pool.imap_unordered(run_one, todo):
                print(line, flush=True)
    rows = []
    for p in P:
        fp = TMP / f"nc1_{p['id']}.json"
        if fp.exists():
            r = json.loads(fp.read_text(encoding="utf-8"))
            r["rounds"] = ROUNDS.get(r["id"], 1); r["self_eval"] = NOTES.get(r["id"], "")
            rows.append(r)
    (OUT / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
