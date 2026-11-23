# CS182 paper: supplementary experiments (2026-11-23)

## 1. Physics validity (k=0, local Gurobi re-check; Wilson 95% interval in brackets)
Full table (including the connectivity-check criterion and Fisher tests): `stats_table.md`, `stats.json`; raw data in `gurobi_all.json`.

| Run | Original 90 prompts | New-category 28 prompts |
|---|---|---|
| Base | 0/90 [0%, 4%] | 0/28 |
| A_s0 (4000 human) | 15/90 [9%, 25%] | 3/28 |
| B_s0 (A + 3x AI) | 26/90 [20%, 39%] | 1/28 |
| B_s1 | 20/90 [15%, 32%] | 4/28 |
| C_s0 (A + 3x extra human) | 21/90 [16%, 33%] | 4/28 |

**Fisher tests**
- A vs base: p = 3e-5.
- B_s0 vs A: two-sided p = 0.075, one-sided p = 0.037.
- B_s1 vs A: p = 0.45; C vs A: p = 0.35.
- B_s0 vs C: p = 0.50; B_s1 vs C: p = 1.0.
- Between the two B seeds: 26 vs 20, p = 0.39.

**Conclusion**: no detectable difference between AI data and the same amount of human data (C).

**Correction**: A_s0 is 15, not the previously reported 14; B_s0 is 26. The older re-check had one Gurobi `checker_error` each, caused by a saturated CPU. `checker_error` is now retried up to 3 times (recorded in the research log).

## 2. Blind recognition (protocol of PREREGISTRATION §4l; judge blind to group; directory `recog/`)
- **Original 90 prompts**: 82 valid structures across A, B_s0, B_s1, C; top-1 recognized **0/82**, top-2 1/82.
- **New categories**: 12 valid structures; top-1 1/12 (one lamp from C).
- **Primary metric** ("valid and recognized" / 28): A 0, B_s0 0, B_s1 0, C 1. All groups are within base + 2, so by §4l the verdict is "new categories not learnable at this scale".
- Valid structures are almost all 2–4 parallel vertical pillars, occasionally a wall slab or a solid block. The physics checker rewards only stability, so the model learned the cheapest stable solution.
- Caveat: a single judge (Claude), and the valid outputs are highly homogeneous.

## 3. Training curves
`train_curves.png`: A_s0, B_s0, B_s1, C_s0 on one plot, all 304 steps.

## 4. GPU part (Kaggle)
- Validation loss on 200 held-out examples for the A_s0, B_s0, B_s1, C_s0 adapters and the base model.
- Train A seed 1 (same config as A_s0, 304 steps); evaluate on the 90 prompts (k=0) and the 28 new-category prompts.
- Held-out set: `data/mix/heldout200.jsonl`, drawn from the training split with seed 4242, disjoint from all training groups and evaluation prompts (checked).
