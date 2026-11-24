"""newcat_0 generator: airplane / house (outside the 21 dataset categories), caption-only parametric designs,
tiled with batch_0.tile2 (support-aware), checked (Gurobi) + filtered + rendered.
python data/opus_gen/newcat_0/gen.py [id ...|merge]
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import types  # noqa: E402
try:
    import scipy  # noqa: F401
except ImportError:
    sys.modules["distill_shape"] = types.SimpleNamespace(aligned_iou=None)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("b0gen", ROOT / "data/opus_gen/batch_0/gen.py")
B0 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B0)
tile2, rm = B0.tile2, B0.rm
from opus_pilot import box, render_png  # noqa: E402
from brickexp.checker import check  # noqa: E402
from distill_filter import feats, keep  # noqa: E402

OUT = Path(__file__).resolve().parent


# ------------------------------------------------ airplanes (y = length, nose at high y; x = span)
def plane(L=16, y0=2, fw=2, fh=2, z0=0, wy=8, wc=3, span=16, wz=None, delta=False, sweep=0, tail_t=False,
          fin=3, stab=6, cockpit=False, engines=False, bi=False, floats=False, gear=False, nose=False, xc=10):
    V = set()
    if gear or floats:
        z0 = 2
    fx0 = xc - fw // 2
    box(V, fx0, fx0 + fw, y0, y0 + L, z0, z0 + fh)                       # fuselage
    if nose:
        box(V, xc - 1, xc + 1, y0 + L, y0 + L + 2, z0, z0 + 1)
    if cockpit:
        box(V, fx0, fx0 + fw, y0 + L - 5, y0 + L - 1, z0 + fh, z0 + fh + 1)
    wz = z0 if wz is None else wz
    if floats:
        wz = max(wz, z0)
    wings = [wz] + ([wz + 4] if bi else [])
    for zz in wings:
        for z in (zz, zz + 1):                                              # double staggered wing layer
            if delta:
                for k in range(wc):
                    half = max(fw // 2 + 1, span // 2 - (k * (span // 2)) // wc)
                    box(V, xc - half, xc + half, wy + k, wy + k + 1, z, z + 1)
            else:
                for dx in range(span // 2):
                    off = (dx * sweep) // (span // 2) if sweep else 0
                    for x in (xc + dx, xc - 1 - dx):
                        box(V, x, x + 1, wy - off, wy - off + wc, z, z + 1)
    if bi:   # struts between wings + fuselage column
        for sx in (xc - span // 2 + 1, xc + span // 2 - 2):
            box(V, sx, sx + 1, wy, wy + 1, wz + 2, wz + 4)
        box(V, fx0, fx0 + fw, wy, wy + wc, z0 + fh, wz + 4)
    ty, sz = y0, z0 + fh
    sb = (sz + fin - 1) if tail_t else max(z0, sz - 1)
    if stab > fw:   # two staggered layers
        s0 = sb if tail_t or fh < 2 else sb - 1
        box(V, xc - stab // 2, xc + stab // 2, ty, ty + 2, s0, s0 + 2)
    box(V, xc - 1, xc, ty, ty + 3, sz, sz + fin)                          # fin
    if engines:
        for ex in (xc - span // 4 - 2, xc + span // 4):
            box(V, ex, ex + 2, wy - 1, wy + 3, wz - 1, wz)
            box(V, ex, ex + 2, wy, wy + 2, 0, wz - 1)
    if gear:
        for gx, gy in ((xc - 4, wy), (xc + 3, wy), (xc - 1, y0 + L - 2)):
            box(V, gx, gx + 1, gy, gy + 2, 0, z0)
    if floats:
        for fxx in (xc - span // 4 - 2, xc + span // 4):
            box(V, fxx, fxx + 2, y0 + 1, wy + 6, 0, 1)
            box(V, fxx, fxx + 2, wy, wy + wc, 1, wz)
        for by in (y0 + 1, wy + 5):   # spreader bars carrying the fuselage
            box(V, xc - span // 4 - 2, xc + span // 4 + 2, by, by + 1, 1, 2)
            box(V, fx0, fx0 + fw, by, by + 1, 1, z0)
    return V


# ------------------------------------------------ houses
def roof_gable(V, x0, x1, y0, y1, z, axis="x", hip=False, flat=False):
    if flat:
        box(V, x0, x1, y0, y1, z, z + 1); box(V, x0, x1, y0, y1, z + 1, z + 2, hollow=True); return
    k = 0
    while True:
        if axis == "x":    # ridge runs along x
            a, b = y0 + k, y1 - k; c, d = (x0 + k, x1 - k) if hip else (x0, x1)
            if b - a < 1 or d - c < 1: break
            box(V, c, d, a, b, z + k, z + k + 1)
            if not hip:
                box(V, x0, x0 + 1, a, b, z + k, z + k + 1); box(V, x1 - 1, x1, a, b, z + k, z + k + 1)
        else:
            a, b = x0 + k, x1 - k; c, d = (y0 + k, y1 - k) if hip else (y0, y1)
            if b - a < 1 or d - c < 1: break
            box(V, a, b, c, d, z + k, z + k + 1)
            if not hip:
                box(V, a, b, y0, y0 + 1, z + k, z + k + 1); box(V, a, b, y1 - 1, y1, z + k, z + k + 1)
        k += 1
    return z + k


def walls(V, x0, x1, y0, y1, z0, z1, door=None, win=(), floor=True):
    if floor:
        box(V, x0, x1, y0, y1, z0, z0 + (2 if z0 > 0 else 1))
    box(V, x0, x1, y0, y1, z0, z1, hollow=True)
    box(V, x0, x1, y0, y1, z1 - 1, z1)             # ceiling slab ties the walls
    if door:
        dx, dw, dh = door
        rm(V, dx, dx + dw, y0, y0 + 1, z0 + 1, z0 + 1 + dh)
    for side, p, zw in win:
        if side == "f": rm(V, p, p + 2, y0, y0 + 1, zw, zw + 1)
        if side == "b": rm(V, p, p + 2, y1 - 1, y1, zw, zw + 1)
        if side == "l": rm(V, x0, x0 + 1, p, p + 2, zw, zw + 1)
        if side == "r": rm(V, x1 - 1, x1, p, p + 2, zw, zw + 1)


def house(x0=5, x1=15, y0=5, y1=13, H=5, roof="gable", axis="x", door=(9, 2, 3), win=(), z0=0, floors=1):
    V = set(); z = z0
    for f in range(floors):
        ws = [(w[0], w[1], z + w[2]) for w in win if len(w) < 4 or w[3] == f]
        walls(V, x0, x1, y0, y1, z, z + H, door=door if f == 0 else None, win=ws, floor=(f == 0))
        z += H
    if roof == "gable": roof_gable(V, x0, x1, y0, y1, z, axis)
    elif roof == "hip": roof_gable(V, x0, x1, y0, y1, z, axis, hip=True)
    elif roof == "flat": roof_gable(V, x0, x1, y0, y1, z, flat=True)
    return V


A = lambda **k: (lambda r: plane(**k))
Hs = lambda **k: (lambda r: house(**k))


def stilts_house():
    V = house(x0=4, x1=14, y0=7, y1=15, H=5, z0=3, door=(6, 2, 3), win=[("f", 10, 2), ("l", 10, 2), ("r", 10, 2)])
    for lx, ly in ((4, 7), (12, 7), (4, 13), (12, 13)):
        box(V, lx, lx + 2, ly, ly + 2, 0, 3)
    for s in range(3):   # stairs up to the door
        box(V, 6, 8, 4 + s, 5 + s, 0, s + 1)
    return V


def garage_house():
    V = house(x0=2, x1=10, y0=5, y1=13, H=5, door=(5, 2, 3), win=[("l", 8, 2)], axis="y")
    G = set(); walls(G, 10, 18, 5, 12, 0, 4); rm(G, 11, 17, 5, 6, 1, 3)
    box(G, 10, 18, 5, 12, 4, 5)
    return V | G


def lshape():
    V = house(x0=2, x1=12, y0=3, y1=10, H=5, door=(6, 2, 3), win=[("f", 3, 2), ("b", 8, 2)], axis="x")
    W = house(x0=2, x1=9, y0=10, y1=18, H=5, door=None, win=[("l", 13, 2), ("b", 4, 2)], axis="y")
    return V | W


def porch():
    V = house(x0=5, x1=15, y0=8, y1=16, H=5, door=(9, 2, 3), win=[("f", 6, 2), ("f", 12, 2)], axis="x")
    for px in (5, 13):
        box(V, px, px + 2, 4, 6, 1, 4)
    box(V, 5, 15, 4, 8, 4, 5); box(V, 5, 15, 4, 9, 5, 6)   # porch roof (2 staggered layers)
    box(V, 5, 15, 4, 8, 0, 1)                             # porch deck
    return V


def balcony():
    V = house(x0=5, x1=15, y0=7, y1=14, H=4, floors=2, door=(9, 2, 3),
              win=[("f", 6, 2, 0), ("b", 9, 2, 0), ("b", 9, 2, 1), ("f", 12, 2, 0)], axis="x")
    rm(V, 9, 11, 7, 8, 5, 7)                              # balcony door upstairs
    box(V, 6, 14, 4, 8, 3, 5)                             # balcony slab (2 layers, into the wall)
    box(V, 6, 14, 4, 5, 5, 6); box(V, 6, 7, 4, 7, 5, 6); box(V, 13, 14, 4, 7, 5, 6)   # railing
    for px in (6, 12): box(V, px, px + 2, 4, 6, 0, 3)     # posts
    return V


def barn():
    return house(x0=4, x1=16, y0=3, y1=17, H=6, door=(7, 6, 4), win=[("l", 8, 3), ("r", 8, 3), ("l", 12, 3), ("r", 12, 3)], axis="y")


def townhouse():
    V = house(x0=7, x1=13, y0=6, y1=13, H=4, floors=3, door=(9, 2, 3), roof=None,
              win=[("f", 8, 2, 1), ("f", 10, 2, 1), ("f", 8, 2, 2), ("f", 10, 2, 2), ("b", 9, 2, 0), ("b", 9, 2, 1)])
    for k, (a, b) in enumerate([(7, 13), (8, 12), (9, 11)]):   # stepped (crow-step) roof
        box(V, a, b, 6, 13, 12 + k, 13 + k)
        if k == 0: rm(V, 8, 12, 7, 12, 12, 13)
    return V


def cottage():
    V = house(x0=4, x1=16, y0=5, y1=14, H=5, door=(9, 2, 3), win=[("f", 5, 2), ("f", 13, 2), ("b", 9, 2)], axis="x")
    box(V, 12, 14, 11, 13, 4, 11)    # chimney rising through the roof
    return V


def glass_box():
    V = set(); box(V, 4, 16, 5, 14, 0, 1)
    box(V, 4, 16, 5, 14, 1, 6, hollow=True)
    rm(V, 5, 15, 5, 6, 1, 5)                               # big glass front: open frame
    box(V, 8, 9, 5, 6, 1, 5); box(V, 11, 12, 5, 6, 1, 5)   # mullions
    box(V, 4, 16, 5, 14, 5, 6)
    box(V, 3, 17, 4, 15, 6, 7); box(V, 4, 16, 5, 14, 7, 8, hollow=True)                             # overhanging flat roof
    return V


DESIGNS = {
    "airplane_03": A(L=12, y0=4, span=12, wy=10, wc=2, fin=2, stab=6),
    "airplane_12": A(L=12, y0=4, span=12, wy=10, wc=2, fin=3, stab=6, tail_t=True),
    "airplane_14": A(L=19, y0=0, fw=3, fh=3, span=18, wy=8, wc=3, fin=4, stab=8, cockpit=True),
    "airplane_07": A(L=14, y0=3, fw=2, fh=2, span=16, wy=9, wc=3, fin=5, stab=6),
    "airplane_11": A(L=16, y0=2, fw=2, fh=2, z0=2, span=16, wy=8, wc=3, fin=4, stab=6, engines=True),
    "airplane_04": A(L=14, y0=2, fw=2, fh=2, span=14, wy=5, wc=7, delta=True, fin=3, stab=2, nose=True),
    "airplane_13": A(L=9, y0=5, fw=4, fh=3, span=12, wy=9, wc=3, fin=3, stab=6),
    "airplane_09": A(L=12, y0=4, span=12, wy=9, wc=2, fin=3, stab=6, gear=True),
    "airplane_08": A(L=16, y0=2, fw=2, fh=1, span=20, wy=10, wc=2, fin=3, stab=6),
    "airplane_01": A(L=16, y0=2, fw=2, fh=2, span=18, wy=10, wc=3, sweep=4, fin=4, stab=6),
    "airplane_00": A(L=12, y0=4, span=12, wy=10, wc=2, fin=3, stab=4),
    "airplane_10": A(L=13, y0=3, span=14, wy=9, wc=3, fin=3, stab=6, floats=True),
    "airplane_06": A(L=11, y0=4, span=12, wy=10, wc=2, fin=2, stab=4, bi=True),
    "airplane_02": A(L=11, y0=4, fw=4, fh=3, span=18, wy=9, wc=3, wz=1, fin=4, stab=8),
    "airplane_05": A(L=17, y0=1, fw=6, fh=4, span=20, wy=7, wc=4, wz=2, fin=5, stab=8),
    "house_11": lambda r: barn(),
    "house_05": lambda r: townhouse(),
    "house_02": lambda r: cottage(),
    "house_10": lambda r: balcony(),
    "house_07": lambda r: garage_house(),
    "house_04": lambda r: porch(),
    "house_09": Hs(x0=7, x1=13, y0=7, y1=13, H=4, door=(9, 2, 2), roof="hip"),
    "house_00": Hs(x0=6, x1=14, y0=7, y1=13, H=4, door=(9, 2, 2), axis="x"),
    "house_13": lambda r: glass_box(),
    "house_08": lambda r: lshape(),
    "house_12": Hs(x0=6, x1=14, y0=6, y1=14, H=4, door=(9, 2, 2), roof="hip",
                   win=[("f", 6, 2), ("b", 9, 2), ("l", 9, 2), ("r", 9, 2)]),
    "house_03": Hs(x0=6, x1=14, y0=6, y1=14, H=6, door=(9, 2, 3), roof="flat", win=[("f", 11, 3)]),
    "house_01": Hs(x0=5, x1=15, y0=6, y1=13, H=4, floors=2, door=(9, 2, 2), axis="x",
                   win=[("f", 6, 2, 0), ("f", 12, 2, 0), ("f", 6, 2, 1), ("f", 12, 2, 1)]),
    "house_14": lambda r: stilts_house(),
    "house_06": Hs(x0=5, x1=15, y0=7, y1=13, H=4, door=(6, 2, 2), axis="x", win=[("f", 11, 2)]),
}
ROUNDS = {'airplane_14': 2, 'airplane_02': 2, 'airplane_05': 3, 'airplane_10': 3, 'house_09': 2, 'house_02': 2, 'house_10': 2, 'house_11': 2, 'house_12': 2, 'house_13': 3, 'house_01': 2, 'house_14': 3, 'house_08': 2}
NOTES = {'airplane_00': '机身+双层交错机翼+尾翼，十字剪影可辨，像飞机', 'airplane_01': '机身+双层交错机翼+尾翼，十字剪影可辨，像飞机', 'airplane_02': '机身+双层交错机翼+尾翼，十字剪影可辨，像飞机', 'airplane_03': '机身+双层交错机翼+尾翼，十字剪影可辨，像飞机', 'airplane_04': '三角翼逐行收窄+尖鼻，像战斗机', 'airplane_05': '宽胖机身+上单翼，像运输机', 'airplane_06': '上下两层机翼+支柱，像双翼机', 'airplane_07': '机身+双层交错机翼+尾翼，十字剪影可辨，像飞机', 'airplane_08': '机身+双层交错机翼+尾翼，十字剪影可辨，像飞机', 'airplane_09': '三点起落架架高机身，像停放的飞机', 'airplane_10': '两条浮筒+撑杆托起机身机翼，像水上飞机', 'airplane_11': '翼下两台发动机（落地支撑），像小喷气机', 'airplane_12': '机身+双层交错机翼+尾翼，十字剪影可辨，像飞机', 'airplane_13': '机身+双层交错机翼+尾翼，十字剪影可辨，像飞机', 'airplane_14': '机身+双层交错机翼+尾翼，十字剪影可辨，像飞机', 'house_00': '空心墙体+门窗+阶梯坡屋顶，像房子', 'house_01': '空心墙体+门窗+阶梯坡屋顶，像房子', 'house_02': '坡屋顶+穿顶烟囱，像小屋', 'house_03': '立方体+平顶带矮墙，像方盒子房', 'house_04': '前廊柱+廊顶+三角屋顶，可辨', 'house_05': '三层窄高+阶梯形顶，像联排屋', 'house_06': '空心墙体+门窗+阶梯坡屋顶，像房子', 'house_07': '主屋+侧边平顶车库带大门洞，可辨', 'house_08': '两翼成L形各自山墙顶，可辨', 'house_09': '小屋+四坡尖顶，像小棚屋', 'house_10': '二层挑出阳台带栏杆，可辨', 'house_11': '大门洞+长坡屋顶，像谷仓', 'house_12': '空心墙体+门窗+阶梯坡屋顶，像房子', 'house_13': '大玻璃立面开口+平顶；挑檐悬空3轮未通过物理检查', 'house_14': '四根高脚柱+台阶通向门，像高脚屋'}


def run_one(p):
    i = p["id"]
    V = {v for v in DESIGNS[i](0) if all(0 <= c < 20 for c in v)}
    text = tile2(V); f = feats(text)
    r = check(text, use_gurobi=True, gurobi_kwargs={"time_limit": 400, "threads": 2}).to_dict()
    render_png(text, OUT / "renders" / f"{i}.png")
    row = {"id": i, "newcat": p["newcat"], "caption": p["caption"], "bucket": p["bucket"], "bricks": text,
           "physics_ok": r["ok"], "filter_ok": bool(r["ok"] and keep(f)),
           "rounds": ROUNDS.get(i, 1), "n_bricks": f["n_bricks"], "self_eval": NOTES.get(i, "")}
    (OUT / "tmp").mkdir(exist_ok=True)
    (OUT / "tmp" / f"{i}.json").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return f"{i} {p['bucket']} n={f['n_bricks']} ok={r['ok']} keep={keep(f)} fill={f['fill']:.2f} {r['message'][:80]}"


def main():
    from multiprocessing import Pool
    only = sys.argv[1:]
    P = [json.loads(l) for l in open(ROOT / "data/opus_gen/newcat_prompts_0.jsonl", encoding="utf-8")]
    (OUT / "renders").mkdir(exist_ok=True)
    todo = [p for p in P if not only or p["id"] in only]
    if only != ["merge"]:
        with Pool(int(os.environ.get("NP", "8"))) as pool:
            for line in pool.imap_unordered(run_one, todo):
                print(line, flush=True)
    rows = [json.loads((OUT / "tmp" / f"{p['id']}.json").read_text(encoding="utf-8"))
            for p in P if (OUT / "tmp" / f"{p['id']}.json").exists()]
    for r in rows:
        r["rounds"] = ROUNDS.get(r["id"], 1); r["self_eval"] = NOTES.get(r["id"], "")
    (OUT / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
