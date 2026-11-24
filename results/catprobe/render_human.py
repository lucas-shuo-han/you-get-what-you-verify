import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import types
sys.modules.setdefault("distill_shape", types.SimpleNamespace(aligned_iou=None))
from opus_pilot import render_png
from brickexp.data import load_split
sys.path.insert(0, str(ROOT / "results/recognition_eval"))
CAT = {"02801938":"basket","02818832":"bed","02828884":"bench","02843684":"birdhouse","02871439":"bookshelf","02876657":"bottle","02880940":"bowl","02924116":"bus","02942699":"camera","02958343":"car","03001627":"chair","03467517":"guitar","03593526":"jar","03797390":"mug","03928116":"piano","03991062":"pot","04256520":"sofa","04379243":"table","04460130":"tower","04468005":"train","04530566":"vessel"}
rng = random.Random(1122); by = {}
for r in load_split("train"): by.setdefault(CAT[r["category_id"]], []).append(r)
out = ROOT / "results/catprobe/items_human.jsonl"; R = ROOT / "results/catprobe/renders/human"; R.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for c, rs in sorted(by.items()):
        for r in rng.sample(rs, 8):
            p = R / f"{c}_{r['structure_id'][:8]}.png"; render_png(r["bricks"], p)
            f.write(json.dumps({"group": "human", "label": c, "render": str(p)}) + "\n")
print("done")
