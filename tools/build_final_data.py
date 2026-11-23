"""Final experiment data (§4n): A = 4000 human (same as data/mix/A.jsonl);
B = A + N Gemini structures (physics_ok & filter_ok, one per structure_id);
C = A + the HUMAN answers for the SAME N structure_ids (paired control: same prompts, only the answer source differs).
Writes data/mix/FA.jsonl FB.jsonl FC.jsonl + data/mix/final_manifest.json."""
import json, glob, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from brickexp import prompts
from brickexp.data import load_split, get_eval_set
def row(cap, bricks, src, sid):
    return {"messages": prompts.initial_messages(cap) + [{"role": "assistant", "content": bricks.strip()}], "src": src, "structure_id": sid}
A = [json.loads(l) for l in open(ROOT / "data/mix/A.jsonl", encoding="utf-8")]
aid = {r["structure_id"] for r in A}; ev = {r["structure_id"] for r in get_eval_set(30)}
train = {r["structure_id"]: r for r in load_split("train")}
gem = {}
for f in sorted(glob.glob(str(ROOT / "data/gemini_gen/final/s0*/samples.jsonl"))):
    for l in open(f, encoding="utf-8"):
        try: r = json.loads(l)
        except Exception: continue
        sid = r.get("structure_id")
        if r.get("physics_ok") and r.get("filter_ok") and sid in train and sid not in aid and sid not in ev and sid not in gem:
            gem[sid] = r
sids = sorted(gem)
B = A + [row(train[s]["caption"], gem[s]["bricks"], "gemini", s) for s in sids]
C = A + [row(train[s]["caption"], train[s]["bricks"], "human_paired", s) for s in sids]
for name, rows in (("FA", A), ("FB", B), ("FC", C)):
    rows = list(rows); random.Random(7).shuffle(rows)
    (ROOT / f"data/mix/{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
tok = lambda rows: sum(len(r["messages"][-1]["content"]) for r in rows)
man = {"FA": len(A), "FB": len(B), "FC": len(C), "n_ai": len(sids), "steps": -(-len(B) // 16),
       "answer_chars_extra": {"gemini": tok(B) - tok(A), "human_paired": tok(C) - tok(A)}}
(ROOT / "data/mix/final_manifest.json").write_text(json.dumps(man, indent=1)); print(man)
