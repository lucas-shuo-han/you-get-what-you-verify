"""Verify a run by its OUTPUT: eval jsonl has 420 records, each with >=1 attempt (+ raw field), summary has eval curve,
and per-brick post-filter JSON can be computed and written. Exit 1 loudly otherwise."""
import json, subprocess, sys
from pathlib import Path
d, fname = Path(sys.argv[1]), sys.argv[2]          # run dir, e.g. eval.jsonl / eval_raw.jsonl
p = d / fname
try:
    recs = [json.loads(l) for l in open(p)]
    assert len(recs) == 420, f"{len(recs)} records"
    assert all(len(r["attempts"]) >= 1 for r in recs), "record without attempts"
    assert all("raw" in a for r in recs for a in r["attempts"]), "missing raw"
    s = json.load(open(d / "train_log.json"))["summary"]
    assert any(k.startswith("eval_curve") for k in s), "no eval_curve in train_log summary"
    out = d / f"per_brick_postfilter_k0_{fname.replace('.jsonl','')}.json"
    subprocess.run([sys.executable, "-m", "brickexp.tiny.per_brick_metrics", str(p), "--post_filter", "--k", "0",
                    "--boot", "2000", "--out", str(out)], check=True, capture_output=True, cwd="/content/RLtheLEGO")
    assert json.load(open(out))[0]["n_prompts"] == 420, "per-brick n_prompts != 420"
    print(f"CHECK_OK {p} n=420 attempts={sum(len(r['attempts']) for r in recs)}")
except Exception as e:
    print(f"CHECK_FAIL {p}: {e!r}"); sys.exit(1)
