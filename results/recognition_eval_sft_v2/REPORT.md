# Blind recognition of valid distilled-SFT structures (v2, clean re-run)

**Why re-run**: the v1 directory (`results/recognition_eval_sft/`) was written concurrently by another session's sub-agent: `blind/` contained 56 images from two runs, the mapping was overwritten 5 s after judgments were written, and ids no longer matched. v1 is void.

**Protocol**: identical to `results/recognition_eval` — fixed 0–20 axes, two views, random blind ids, 6 per sheet. The judge (Claude) sees only the sheets and picks top-1/top-2 among 21 classes. The mapping is read only after all judgments are written to `judgments.jsonl`. Consistency check: blind images, judgments and mapping ids match exactly (asserted in `build.py`).

**Samples**
- sft: the 14 Gurobi-valid structures produced by the distilled-SFT model on the evaluation set.
- human: for each sft structure, one human structure of the same category drawn at random from the training set (`build.py`, seed 20261121).

| Group | n | top-1 | top-2 |
|---|---|---|---|
| human (same-category) | 14 | 12/14 = 86% | 13/14 = 93% |
| sft (valid distilled SFT) | 14 | **0/14** | **0/14** |

The sft structures were judged tower 6 times, otherwise bookshelf, bench, bus, jar, piano, sofa, bed — **none correct**.

**Caveats**: the judge had seen the same sft renders in v1 but without labels, so there is no label leakage. Single same-family judge and small n, but 0/14 vs 12/14 is far outside noise.
