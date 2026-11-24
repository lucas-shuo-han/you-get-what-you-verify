"""Shape comparison of the Gemini paired data (FB vs FC extras: same 665 prompts, Gemini vs human answers).
Aspect ratio = longest / shortest bounding-box extent (x, y, z in grid units).
Reports median aspect ratio per source and the number of pairs whose ratios are within 30% of each other.
Usage: python tools/build_final_data.py && python tools/gemini_pair_shape.py"""
import json, re, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
pat = re.compile(r"(\d+)x(\d+) \((\d+),(\d+),(\d+)\)")
def extents(text):
    xs, ys, zs = [], [], []
    for h, w, x, y, z in (map(int, m) for m in pat.findall(text)):
        xs += [x, x + h]; ys += [y, y + w]; zs += [z, z + 1]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
def load(name, src):
    out = {}
    for l in open(ROOT / f"data/mix/{name}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["src"] == src: out[r["structure_id"]] = extents(r["messages"][-1]["content"])
    return out
g, h = load("FB", "gemini"), load("FC", "human_paired")
ids = sorted(set(g) & set(h))
ar = lambda d: max(d) / max(1, min(d))
a = [ar(g[i]) for i in ids]; b = [ar(h[i]) for i in ids]
res = {"n_pairs": len(ids), "definition": "longest/shortest bbox extent",
       "median_aspect_gemini": round(statistics.median(a), 2), "median_aspect_human": round(statistics.median(b), 2),
       "pairs_within_30pct": sum(abs(x - y) <= 0.3 * max(x, y) for x, y in zip(a, b))}
print(json.dumps(res, indent=1))
(ROOT / "results/gemini_pairs_shape.json").write_text(json.dumps(res, indent=1))
