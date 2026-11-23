"""Blind recognition eval for the 14 physically-valid distill-SFT structures + 14 same-category human controls.
Protocol identical to results/recognition_eval/build.py (fixed 0..20 axes, shuffled, random blind ids)."""
import json, random, shutil, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from opus_pilot import render_png
from brickexp.data import load_split
sys.path.insert(0, str(ROOT / "results/recognition_eval"))
CAT = {"02801938":"basket","02818832":"bed","02828884":"bench","02843684":"birdhouse","02871439":"bookshelf",
"02876657":"bottle","02880940":"bowl","02924116":"bus","02942699":"camera","02958343":"car","03001627":"chair",
"03467517":"guitar","03593526":"jar","03797390":"mug","03928116":"piano","03991062":"pot","04256520":"sofa",
"04379243":"table","04460130":"tower","04468005":"train","04530566":"vessel"}
OUT = ROOT / "results/recognition_eval_sft"
rng = random.Random(20261121)
recs = {}
for sp in ("train", "test"):
    try:
        for r in load_split(sp): recs.setdefault(r["structure_id"], r)
    except Exception as ex: print("split", sp, ex)
train = load_split("train")
items = []  # (group, sid, label, text, solved_at)
for l in open(ROOT / "results/distill_sft/runs/eval_sft.jsonl"):
    e = json.loads(l)
    if e["solved_at"] is None: continue
    a = next(a for a in e["attempts"] if a["attempt"] == e["solved_at"]) if any(a["attempt"] == e["solved_at"] for a in e["attempts"]) else e["attempts"][-1]
    assert a["ok"]
    lab = CAT[recs[e["structure_id"]]["category_id"]]
    items.append(("sft", e["structure_id"], lab, a["text"], e["solved_at"]))
assert len(items) == 14, len(items)
used = {i[1] for i in items}
for g, sid, lab, _, _ in list(items):
    pool = [r for r in train if CAT[r["category_id"]] == lab and r["structure_id"] not in used]
    r = rng.choice(pool); used.add(r["structure_id"])
    items.append(("human", r["structure_id"], lab, r["bricks"], None))
rng.shuffle(items); mapping = []; (OUT / "blind").mkdir(parents=True, exist_ok=True)
for g, sid, lab, text, k in items:
    p = OUT / "renders" / g / f"{lab}_{sid[:8]}.png"; p.parent.mkdir(parents=True, exist_ok=True)
    render_png(text, p)
    bid = f"img_{rng.randrange(10**6):06d}"; shutil.copy(p, OUT / "blind" / f"{bid}.png")
    types = Counter(ln.split()[0] for ln in text.strip().splitlines() if ln.strip())
    mapping.append({"blind_id": bid, "group": g, "structure_id": sid, "label": lab, "solved_at": k,
                    "n_bricks": sum(types.values()), "brick_types": dict(types), "render": str(p.relative_to(OUT)),
                    "caption": recs[sid]["caption"]})
json.dump(mapping, open(OUT / "mapping_DO_NOT_READ.json", "w"), indent=1)
print(len(mapping))
