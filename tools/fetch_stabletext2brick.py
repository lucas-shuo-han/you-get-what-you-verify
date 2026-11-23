"""Download StableText2Brick (Hugging Face: AvaLovelace/StableText2Brick) into data/hf_cache.

python tools/fetch_stabletext2brick.py
All scripts that need the dataset call brickexp.data.load_split(), which reads this cache."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brickexp.data import load_split, get_eval_set  # noqa: E402

for split in ("train", "test"):
    print(split, len(load_split(split)), "records")
print("90-prompt eval set:", len(get_eval_set(per_bucket=30)), "prompts")
