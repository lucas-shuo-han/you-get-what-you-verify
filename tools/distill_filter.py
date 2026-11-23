"""Post-hoc quality filter for distilled samples (thresholds fixed BEFORE the 100-prompt run; see REPORT.md).

keep iff verify ok AND
  n_types >= 3                       (distinct brick dimensions, 2x4 and 4x2 counted separately as BrickGPT does)
  max_type_share <= 0.70             (no single brick type > 70% of bricks)
  fill <= 0.85                       (occupied voxels / bounding-box voxels; a solid block has fill ~1)
  n_bricks >= 10
python tools/distill_filter.py results/distill_pilot/train100.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brickexp.checker import parse  # noqa: E402

TH = {"n_types": 3, "max_type_share": 0.70, "fill": 0.85, "n_bricks": 10}


def feats(text):
    b = parse(text)[0]
    types = Counter(f"{x.h}x{x.w}" for x in b)
    vox = sum(x.h * x.w for x in b)
    xs = [x.x for x in b] + [x.x + x.h for x in b]; ys = [x.y for x in b] + [x.y + x.w for x in b]; zs = [x.z for x in b]
    box = (max(xs) - min(xs)) * (max(ys) - min(ys)) * (max(zs) - min(zs) + 1)
    return {"n_bricks": len(b), "n_types": len(types), "max_type_share": max(types.values()) / len(b), "fill": vox / box}


def keep(f):
    return (f["n_types"] >= TH["n_types"] and f["max_type_share"] <= TH["max_type_share"]
            and f["fill"] <= TH["fill"] and f["n_bricks"] >= TH["n_bricks"])


if __name__ == "__main__":
    S = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    rows = []
    for s in S:
        r = {"structure_id": s["structure_id"], "bucket": s["bucket"], "caption": s["caption"], "solved_at": s["solved_at"]}
        if s["solved_at"] is not None:
            t = s["attempts"][s["solved_at"]]["text"]; r.update(feats(t), text=t, keep=keep(feats(t)))
        else:
            r["keep"] = False
        rows.append(r)
    out = Path(sys.argv[1]).with_name(Path(sys.argv[1]).stem + "_filtered.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ok = [r for r in rows if r["solved_at"] is not None]
    print(json.dumps({"n": len(rows), "verify_ok": len(ok), "kept": sum(r["keep"] for r in rows),
                      "kept_oneshot": sum(r["keep"] and r["solved_at"] == 0 for r in rows),
                      "rejected_by": {k: sum(1 for r in ok if not (r[k] >= v if k in ("n_types", "n_bricks") else r[k] <= v))
                                      for k, v in TH.items()},
                      "by_bucket": {b: [sum(r["keep"] for r in rows if r["bucket"] == b), sum(r["bucket"] == b for r in rows)]
                                    for b in ("small", "medium", "large")}}, indent=1))
