"""Aligned voxel IoU (max over x/y integer shifts; STABLE voxel_iou formula) for distilled samples vs dataset targets.
python tools/distill_shape.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brickexp.data import load_split  # noqa: E402
from brickexp.screen.conformal import auc  # noqa: E402
from distill_iou import vox  # noqa: E402

TH = 0.30


def aligned_iou(a, b):
    """max over (dx,dy) shifts of |A∩B| / |A∪B|; intersections for all shifts via 2D cross-correlation per z-layer."""
    na, nb = a.sum(), b.sum()
    inter = sum(fftconvolve(a[:, :, z], b[::-1, ::-1, z], mode="full") for z in range(a.shape[2]))
    inter = np.rint(inter).max()
    return float(inter / (na + nb - inter)) if na + nb - inter > 0 else 0.0


def rows():
    D = Path(__file__).resolve().parent.parent / "results/distill_pilot"; out = []
    for f, set_ in (("smoke3.jsonl", "step1"), ("p20.jsonl", "step1"), ("train100.jsonl", "train100")):
        for l in open(D / f, encoding="utf-8"):
            s = json.loads(l)
            if s["solved_at"] is None:
                continue
            out.append({"set": set_, "structure_id": s["structure_id"], "bucket": s["bucket"], "caption": s["caption"],
                        "text": s["attempts"][s["solved_at"]]["text"]})
    return out


if __name__ == "__main__":
    from distill_filter import feats
    tgt = {r["structure_id"]: r for sp in ("train", "test") for r in load_split(sp)}
    R = rows(); rng = random.Random(0)
    allr = list(tgt.values())
    for r in R:
        f = feats(r["text"]); r["degenerate"] = f["n_types"] == 1 or f["fill"] > 0.85
        g, t = vox(r["text"]), vox(tgt[r["structure_id"]]["bricks"])
        r["iou_raw"] = float((g * t).sum() / np.clip(g + t, 0, 1).sum())
        r["iou_aligned"] = aligned_iou(g, t)
        other = rng.choice([x for x in allr if x["bucket"] == r["bucket"] and x["structure_id"] != r["structure_id"]])
        r["iou_random_human_pair"] = aligned_iou(vox(other["bricks"]), t)
    keep = [r for r in R if r["set"] == "train100" or r["degenerate"]]
    y = np.array([r["degenerate"] for r in keep]); x = np.array([r["iou_aligned"] for r in keep])
    out = {"n": len(keep), "n_degenerate": int(y.sum()),
           "auroc_low_aligned_iou_flags_degenerate": auc(-x, y),
           "aligned_iou_mean": {"degenerate": float(x[y].mean()), "train100": float(x[~y].mean())},
           "frac_ge_th(0.30)": {"degenerate": float((x[y] >= TH).mean()), "train100": float((x[~y] >= TH).mean())},
           "raw_iou_mean": float(np.mean([r["iou_raw"] for r in keep])),
           "random_human_pair_aligned_iou_mean": float(np.mean([r["iou_random_human_pair"] for r in keep])),
           "human_upper_bound": 1.0}
    (Path(__file__).resolve().parent.parent / "results/distill_pilot/shape_rows.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k != "text"}) + "\n" for r in keep))
    print(json.dumps(out, indent=1))
