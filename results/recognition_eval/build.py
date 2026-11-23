"""Recognition-accuracy eval: render groups with fixed 0..20 axes, then blind-shuffle."""
import json, random, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from opus_pilot import render_png
from brickexp.data import load_split
CAT = {"02801938":"basket","02818832":"bed","02828884":"bench","02843684":"birdhouse","02871439":"bookshelf",
"02876657":"bottle","02880940":"bowl","02924116":"bus","02942699":"camera","02958343":"car","03001627":"chair",
"03467517":"guitar","03593526":"jar","03797390":"mug","03928116":"piano","03991062":"pot","04256520":"sofa",
"04379243":"table","04460130":"tower","04468005":"train","04530566":"vessel"}
OUT = ROOT / "results/recognition_eval"
rng = random.Random(2026)
train = load_split("train"); by_id = {r["structure_id"]: r for r in train}
items = []  # (group, sid, text)
for r in rng.sample(train, 20): items.append(("human", r["structure_id"], r["bricks"]))
for l in open(ROOT / "results/harness_vision/episodes.jsonl"):
    e = json.loads(l); items.append(("ds_" + e["arm"], e["structure_id"], next(a["text"] for a in reversed(e["attempts"]) if a.get("text"))))
d = [json.loads(l) for l in open(ROOT / "data/distill/sft_distill.jsonl")]
for e in rng.sample(d, 20): items.append(("ds_distill", e["structure_id"], e["messages"][-1]["content"]))
for l in open(ROOT / "results/opus_pilot/samples.jsonl"):
    e = json.loads(l); items.append(("opus", e["structure_id"], e["text"]))
rng.shuffle(items); mapping = []; (OUT / "blind").mkdir(parents=True, exist_ok=True)
for i, (g, sid, text) in enumerate(items):
    lab = CAT[by_id[sid]["category_id"]]; p = OUT / "renders" / g / f"{lab}_{sid[:8]}.png"; p.parent.mkdir(parents=True, exist_ok=True)
    try: render_png(text, p)
    except Exception as ex: print("render fail", g, sid, ex); continue
    bid = f"img_{rng.randrange(10**6):06d}"; shutil.copy(p, OUT / "blind" / f"{bid}.png")
    mapping.append({"blind_id": bid, "group": g, "structure_id": sid, "label": lab, "render": str(p.relative_to(OUT)), "caption": by_id[sid]["caption"]})
json.dump(mapping, open(OUT / "mapping_DO_NOT_READ.json", "w"), indent=1)
print(len(mapping))
