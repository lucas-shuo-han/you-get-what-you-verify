"""Shape fidelity of distilled samples: voxel IoU vs the dataset target structure (STABLE voxel_iou, no alignment).
python tools/distill_iou.py results/distill_pilot/train100_filtered.jsonl [more filtered/raw jsonl...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "STABLE"))
from brickexp.checker import parse  # noqa: E402
from brickexp.data import load_split  # noqa: E402
from brickexp.screen.conformal import auc  # noqa: E402
from distill_filter import feats  # noqa: E402

TH_IOU = 0.25


def vox(text):
    g = np.zeros((20, 20, 20), np.float32)
    for b in parse(text)[0]:
        g[b.x:b.x + b.h, b.y:b.y + b.w, b.z] = 1
    return g


def voxel_iou(a, b):  # == vendor/STABLE/evaluate_models.py voxel_iou (same 20^3 shape, no padding needed)
    inter = float(np.sum(a * b)); union = float(np.sum(np.clip(a + b, 0, 1)))
    return inter / union if union > 0 else 0.0


if __name__ == "__main__":
    tgt = {}
    for sp in ("train", "test"):
        for r in load_split(sp):
            tgt[r["structure_id"]] = r["bricks"]
    rows = []
    for p in sys.argv[1:]:
        for l in open(p, encoding="utf-8"):
            s = json.loads(l)
            if "attempts" in s:  # raw run file
                if s["solved_at"] is None:
                    continue
                t = s["attempts"][s["solved_at"]]["text"]; f = feats(t)
            else:
                if s["solved_at"] is None:
                    continue
                t = s["text"]; f = {k: s[k] for k in ("n_types", "fill", "max_type_share", "n_bricks")}
            iou = voxel_iou(vox(t), vox(tgt[s["structure_id"]]))
            rows.append({"src": Path(p).stem, "structure_id": s["structure_id"], "bucket": s["bucket"], "iou": iou,
                         "degenerate": f["n_types"] == 1 or f["fill"] > 0.85, **f})
    y = np.array([r["degenerate"] for r in rows]); x = np.array([r["iou"] for r in rows])
    out = {"n": len(rows), "n_degenerate": int(y.sum()), "iou_mean": float(x.mean()) if len(x) else None,
           "iou_mean_degenerate": float(x[y].mean()) if y.any() else None,
           "iou_mean_other": float(x[~y].mean()) if (~y).any() else None,
           "auroc_low_iou_flags_degenerate": auc(-x, y) if 0 < y.sum() < len(y) else None,
           "frac_iou_ge_th": float((x >= TH_IOU).mean()) if len(x) else None}
    Path("results/distill_pilot/iou_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(out, indent=1))
