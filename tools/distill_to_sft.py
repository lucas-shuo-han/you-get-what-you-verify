"""Distilled DeepSeek runs -> SFT chat jsonl ({"messages": [...]}, the format brickexp.train_sft --data_jsonl reads).

Input: run files written by tools/distill_pilot.py (rows with 'attempts' + 'solved_at') and/or their
*_filtered.jsonl companions written by tools/distill_filter.py (rows with 'text' + 'keep').
A sample is used iff verify passed (solved_at is not None) AND (by default) the pre-registered quality filter
(distill_filter.keep) passes. --no_quality drops the quality filter (pipeline smoke only; not for real runs).

prompt  = brickexp.prompts.initial_messages(caption)  -- the plain eval prompt, WITHOUT the --quality diversity
          line used at generation time, so train and eval prompts match exactly.
answer  = the final verified brick lines only (failed attempts / feedback turns are not trained on).
Rows whose structure_id is in the 90-prompt eval set are refused (assert), duplicates (same caption+text) dropped.

python tools/distill_to_sft.py results/distill_pilot/train100.jsonl --out data/distill/sft_distill.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def samples_from_file(path: Path):
    """Yield (structure_id, bucket, caption, text) for every verify-ok row of a run or *_filtered file."""
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("solved_at") is None:
            continue
        if "attempts" in r:  # raw run file
            a = r["attempts"][r["solved_at"]]
            assert a["ok"], f"{path}: solved_at points at a failed attempt ({r['structure_id']})"
            text = a["text"]
        else:  # *_filtered.jsonl
            text = r["text"]
        yield r["structure_id"], r["bucket"], r["caption"], text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="run jsonl and/or *_filtered.jsonl files (read-only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_quality", action="store_true", help="skip the quality filter (smoke tests only)")
    ap.add_argument("--allow_eval_overlap", action="store_true", help="debug only: do not refuse eval-set prompts")
    a = ap.parse_args()

    from brickexp import prompts
    from brickexp.checker import check
    from brickexp.data import get_eval_set
    from distill_filter import feats, keep

    eval_ids = {r["structure_id"] for r in get_eval_set(30)}
    seen, rows, stats = set(), [], {"verify_ok": 0, "quality_rejected": 0, "dup": 0, "eval_overlap": 0, "recheck_fail": 0}
    for p in a.inputs:
        for sid, bucket, cap, text in samples_from_file(Path(p)):
            stats["verify_ok"] += 1
            if sid in eval_ids:
                stats["eval_overlap"] += 1
                if not a.allow_eval_overlap:
                    continue
            # cheap re-check (format/bounds/collision/floating) guards against schema drift; Gurobi is not re-run
            rc = check(text, use_gurobi=False)
            if not rc.ok and rc.failure != "unstable":
                stats["recheck_fail"] += 1
                continue
            if not a.no_quality and not keep(feats(text)):
                stats["quality_rejected"] += 1
                continue
            k = (cap, text)
            if k in seen:
                stats["dup"] += 1
                continue
            seen.add(k)
            rows.append({"messages": prompts.initial_messages(cap) + [{"role": "assistant", "content": text.strip()}],
                         "structure_id": sid, "bucket": bucket})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    stats["written"] = len(rows)
    print(json.dumps(stats), "->", out)
    if stats["eval_overlap"] and not a.allow_eval_overlap:
        print(f"WARNING: skipped {stats['eval_overlap']} samples whose prompt is in the 90-prompt eval set")


if __name__ == "__main__":
    main()
