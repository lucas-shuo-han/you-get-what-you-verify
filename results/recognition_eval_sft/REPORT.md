# Blind recognition of valid distilled-SFT structures (v1 — VOID)

> The files in this directory were written concurrently by two sessions (the mapping was overwritten and `blind/` contains 56 images). The numbers here are void; use `results/recognition_eval_sft_v2/` instead.

Protocol: same as `results/recognition_eval` (fixed 0–20 axes, two views, shuffled blind ids, 6-per-sheet tiling, top-1/top-2 among 21 classes, mapping read only after `judgments.jsonl` was written).

| Group | n | top-1 | top-2 |
|---|---|---|---|
| human (same 14 prompts) | 14 | 10/14 = 71% | 13/14 = 93% |
| sft (valid distilled-SFT structures) | 14 | 0/14 | 0/14 |
