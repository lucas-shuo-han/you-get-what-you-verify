"""Mixed human + AI SFT data (user request: "mix all data sources; data volume comes first").

A = 4000 human (StableText2Brick train, seed 2026, excluding the 90 eval prompts and every structure_id used by AI data)
B = A + 3 x AI, AI = Opus 187 (data/opus_gen/opus_all_ok.jsonl)
               + DeepSeek 65 (data/distill/sft_distill.jsonl rows listed in results/distill_data_audit/keep_ids.json)
               + new-category train half (data/opus_gen/newcat_{0,1}/samples.jsonl, id in newcat_train.jsonl, verify-ok)
Prompts are the plain eval prompt (brickexp.prompts.initial_messages) for every source.
Also writes data/mix/newcat_test_eval.jsonl (eval rows for eval_2x2 --eval_jsonl) and data/mix/manifest.json.
python tools/build_mix_data.py [--allow_no_newcat]
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brickexp import prompts  # noqa: E402
from brickexp.checker import check  # noqa: E402
from brickexp.data import get_eval_set, load_split  # noqa: E402

OUT = ROOT / "data/mix"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def row(cap, bricks, src, sid):
    return {"messages": prompts.initial_messages(cap) + [{"role": "assistant", "content": bricks.strip()}],
            "src": src, "structure_id": sid}


def newcat_train():
    ids = {r["id"] for r in jl(ROOT / "data/opus_gen/newcat_train.jsonl")}
    caps = {r["id"]: r["caption"] for r in jl(ROOT / "data/opus_gen/newcat_train.jsonl")}
    out = []
    for k in (0, 1):
        p = ROOT / f"data/opus_gen/newcat_{k}/samples.jsonl"
        if not p.exists():
            continue
        for r in jl(p):
            if r.get("id") not in ids:
                continue
            text = r.get("bricks") or r.get("text") or ""
            if r.get("ok") is False or not text or not check(text, use_gurobi=False).ok:
                continue  # verify-ok only (cheap re-check; Gurobi status from the generator if present)
            if r.get("gurobi_ok") is False or r.get("stable") is False:
                continue
            if not (r.get("physics_ok", True) and r.get("filter_ok", True)):
                continue
            out.append(row(r.get("caption") or caps[r["id"]], text, "opus_newcat", r["id"]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--allow_no_newcat", action="store_true")
    ap.add_argument("--n_human", type=int, default=4000); ap.add_argument("--up", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ev = {r["structure_id"] for r in get_eval_set(30)}
    opus = [row(r["caption"], r["bricks"], "opus", r["structure_id"]) for r in jl(ROOT / "data/opus_gen/opus_all_ok.jsonl")]
    keep = set(json.load(open(ROOT / "results/distill_data_audit/keep_ids.json"))["keep"])
    ds_all = jl(ROOT / "data/distill/sft_distill.jsonl")
    ds = []
    for i, r in enumerate(ds_all):
        if i in keep:
            ds.append({"messages": prompts.initial_messages(_caption(r)) + [r["messages"][-1]],
                       "src": "deepseek", "structure_id": r["structure_id"]})
    nc = newcat_train()
    if not nc and not a.allow_no_newcat:
        sys.exit("new-category train samples not ready (data/opus_gen/newcat_*/samples.jsonl); use --allow_no_newcat for a dry build")
    ai = opus + ds + nc
    used = {r["structure_id"] for r in ai}
    assert not (used & ev), "AI data overlaps the 90 eval prompts"
    pool = sorted((r for r in load_split("train") if r["structure_id"] not in ev and r["structure_id"] not in used),
                  key=lambda r: r["structure_id"])
    human = [row(r["caption"], r["bricks"], "human", r["structure_id"]) for r in random.Random(2026).sample(pool, a.n_human)]
    A = human; B = human + ai * a.up
    random.Random(7).shuffle(B)
    # C: matched control for B -- same size, same repetition, extra rows are HUMAN instead of AI
    hid = {r["structure_id"] for r in human}
    rest = [r for r in pool if r["structure_id"] not in hid]
    extra = [row(r["caption"], r["bricks"], "human_extra", r["structure_id"]) for r in random.Random(2027).sample(rest, len(ai))]
    C = human + extra * a.up
    random.Random(7).shuffle(C)
    for name, rows in (("A", A), ("B", B), ("C", C)):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    ref = {}  # Opus structures for the test prompts: recognition upper reference only (never trained on)
    for k in (0, 1):
        p = ROOT / f"data/opus_gen/newcat_{k}/samples.jsonl"
        if p.exists():
            for r in jl(p):
                if r.get("physics_ok") and r.get("bricks"):
                    ref[r["id"]] = r["bricks"]
    te = [{"structure_id": r["id"], "caption": r["caption"], "bucket": r["bucket"], "category": r["newcat"], "n_bricks": 0,
           "bricks": ref.get(r["id"], "")} for r in jl(ROOT / "data/opus_gen/newcat_test.jsonl")]
    assert not ({r["structure_id"] for r in nc} & {r["structure_id"] for r in te}), "new-category test leaked into train"
    (OUT / "newcat_test_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in te), encoding="utf-8")
    man = {"A": len(A), "B": len(B), "C": len(C), "human": len(human), "ai": {"opus": len(opus), "deepseek": len(ds), "opus_newcat": len(nc)},
           "upsample": a.up, "newcat_test": len(te), "seed_human": 2026}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man))


def _caption(r):
    """caption back out of the plain eval prompt (DeepSeek rows store the full prompt, not the caption)."""
    u = r["messages"][1]["content"]
    return u.split("### Input:\n", 1)[1].split("\n\nEach line", 1)[0]


if __name__ == "__main__":
    main()
