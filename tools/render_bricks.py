"""Render brick structures as 3D voxel plots (one color per brick).
python tools/render_bricks.py <filtered.jsonl> <out_dir> [k]   -> renders up to k kept samples (one per bucket first)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brickexp.checker import parse  # noqa: E402


def render(text, title, path):
    b = parse(text)[0]
    vox = np.zeros((20, 20, 20), bool); col = np.empty((20, 20, 20, 4))
    cm = plt.get_cmap("tab20")
    for k, x in enumerate(b):
        vox[x.x:x.x + x.h, x.y:x.y + x.w, x.z] = True
        col[x.x:x.x + x.h, x.y:x.y + x.w, x.z] = cm(k % 20)
    fig = plt.figure(figsize=(4.2, 4.2)); ax = fig.add_subplot(projection="3d")
    ax.voxels(vox, facecolors=col, edgecolor=(0, 0, 0, .25), linewidth=.3)
    ax.set_xlim(0, 20); ax.set_ylim(0, 20); ax.set_zlim(0, 20)  # fixed grid, never autoscale to the model
    ax.set_box_aspect((1, 1, 1)); ax.view_init(25, -60); ax.set_axis_off()
    ax.set_title(title, fontsize=8, wrap=True)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    kept = [r for r in rows if r.get("keep")]
    pick = []
    for b in ("small", "medium", "large"):
        pick += [r for r in kept if r["bucket"] == b][:1]
    pick += [r for r in kept if r not in pick]
    for i, r in enumerate(pick[:k]):
        cap = r["caption"][:70] + ("…" if len(r["caption"]) > 70 else "")
        render(r["text"], f"[{r['bucket']}, {r['n_bricks']} bricks, {r['n_types']} sizes, fill {r['fill']:.2f}]\n{cap}",
               out / f"sample_{i}_{r['bucket']}.png")
        print(out / f"sample_{i}_{r['bucket']}.png")
