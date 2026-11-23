"""StableText2Brick loader with size buckets and a fixed, seeded eval subset.

Splits: the dataset's official train/test split is used as-is (fixed).
Buckets (by brick count, thresholds fixed so they never move):
  small  : n <= 60      medium : 61..120      large : n > 120
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HF_NAME = "AvaLovelace/StableText2Brick"
BUCKETS = {"small": (0, 60), "medium": (61, 120), "large": (121, 10**9)}
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "data" / "hf_cache"
SPLIT_DIR = ROOT / "data" / "splits"


def n_bricks(bricks: str) -> int:
    return sum(1 for l in bricks.splitlines() if l.strip())


def bucket_of(n: int) -> str:
    for name, (lo, hi) in BUCKETS.items():
        if lo <= n <= hi:
            return name
    raise ValueError(n)


def _to_record(row: dict, caption_idx: int = 0) -> dict:
    caps = row["captions"]
    if isinstance(caps, str):
        caps = [caps]
    n = n_bricks(row["bricks"])
    return {
        "structure_id": row["structure_id"],
        "category_id": row["category_id"],
        "caption": caps[caption_idx % len(caps)],
        "captions": list(caps),
        "bricks": row["bricks"],
        "n_bricks": n,
        "bucket": bucket_of(n),
    }


def load_split(split: str = "test", cache_dir: str | Path | None = DEFAULT_CACHE) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    ds = ds.remove_columns([c for c in ds.column_names if c == "stability_scores"])
    return [_to_record(r) for r in ds]


def make_eval_subset(records: list[dict], per_bucket: int, seed: int = 0) -> list[dict]:
    """Deterministic: sort by structure_id, then seeded sample per bucket."""
    rng = random.Random(seed)
    out = []
    for b in BUCKETS:
        pool = sorted((r for r in records if r["bucket"] == b), key=lambda r: r["structure_id"])
        out += rng.sample(pool, min(per_bucket, len(pool)))
    return out


def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def get_eval_set(per_bucket: int = 100, seed: int = 0, cache_dir=DEFAULT_CACHE) -> list[dict]:
    """Cached at data/splits/test_eval_{per_bucket}_s{seed}.jsonl so every run sees the same prompts."""
    p = SPLIT_DIR / f"test_eval_{per_bucket}_s{seed}.jsonl"
    if p.exists():
        return load_jsonl(p)
    recs = make_eval_subset(load_split("test", cache_dir), per_bucket, seed)
    save_jsonl(recs, p)
    return recs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_bucket", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    recs = get_eval_set(a.per_bucket, a.seed)
    from collections import Counter
    print(len(recs), Counter(r["bucket"] for r in recs))
