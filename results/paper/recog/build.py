"""§4l blind recognition of every Gurobi-passing k=0 structure of A_s0, B_s0, B_s1, C_s0 (judge blind to group).
Two pools so each keeps its candidate set: pool90 (90 original prompts, 21 classes) and poolnew (28 new-category
prompts, 21 + 4 new classes). Fixed 0..20 axes, 2 views (tools/opus_pilot.render_png), shuffled random blind ids,
6-image sheets labelled only by blind id. Judgments are written before the mapping is read."""
import json, random, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from opus_pilot import render_png
from brickexp.data import load_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.image as mpimg
CAT = {"02801938":"basket","02818832":"bed","02828884":"bench","02843684":"birdhouse","02871439":"bookshelf",
"02876657":"bottle","02880940":"bowl","02924116":"bus","02942699":"camera","02958343":"car","03001627":"chair",
"03467517":"guitar","03593526":"jar","03797390":"mug","03928116":"piano","03991062":"pot","04256520":"sofa",
"04379243":"table","04460130":"tower","04468005":"train","04530566":"vessel"}
OUT = ROOT / "results/paper/recog"; rng = random.Random(1123)
G = json.load(open(ROOT / "results/paper/gurobi_all.json"))
by_id = {r["structure_id"]: r for sp in ("train", "test") for r in load_split(sp)}
newcat = {json.loads(l)["id"]: json.loads(l)["newcat"] for l in open(ROOT / "data/opus_gen/newcat_test.jsonl", encoding="utf-8")}
for pool, suite in (("pool90", "90"), ("poolnew", "newcat")):
    items = []
    for run in ("A_s0", "B_s0", "B_s1", "C_s0"):
        g = G[f"{run}_{suite}"]; rows = {json.loads(l)["structure_id"]: json.loads(l) for l in open(ROOT / g["file"], encoding="utf-8")}
        for sid in g["ok_ids"]:
            lab = newcat[sid] if suite == "newcat" else CAT[by_id[sid]["category_id"]]
            items.append((run, sid, lab, rows[sid]["attempts"][0]["raw"]))
    rng.shuffle(items); M = []; (OUT / pool / "blind").mkdir(parents=True, exist_ok=True)
    for run, sid, lab, text in items:
        p = OUT / pool / "renders" / run / f"{lab}_{sid[:8]}.png"; p.parent.mkdir(parents=True, exist_ok=True)
        render_png(text, p)
        bid = f"img_{rng.randrange(10**6):06d}"; shutil.copy(p, OUT / pool / "blind" / f"{bid}.png")
        M.append({"blind_id": bid, "group": run, "structure_id": sid, "label": lab})
    json.dump(M, open(OUT / pool / "mapping_DO_NOT_READ.json", "w"), indent=1)
    ids = [m["blind_id"] for m in M]; rng.shuffle(ids); (OUT / pool / "sheets").mkdir(exist_ok=True)
    for k in range(0, len(ids), 6):
        fig, axs = plt.subplots(3, 2, figsize=(12, 9))
        for ax, b in zip(axs.flat, ids[k:k + 6] + [None] * 6):
            ax.axis("off")
            if b: ax.imshow(mpimg.imread(OUT / pool / "blind" / f"{b}.png")); ax.set_title(b, fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / pool / "sheets" / f"sheet_{k // 6:02d}.png", dpi=80); plt.close(fig)
    print(pool, len(M), "images", (len(ids) + 5) // 6, "sheets")
