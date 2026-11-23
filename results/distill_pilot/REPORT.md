# DeepSeek black-box distillation pilot

## Step 1 (23 evaluation prompts)
- Model deepseek-chat, temperature 0.6, max_tokens 2048; prompts from `brickexp/prompts.py` (same as Ref and C1).
- Judge: local verify (format, out-of-bounds, collision, floating) + Gurobi stability. Feedback: verify errors returned to the model, history window 2, ≤ 3 rounds.
- Code: `tools/distill_pilot.py`. The API key is read from the environment only at request time; all outputs were checked to contain no key.

| Setting | All | small | medium | large |
|---|---|---|---|---|
| (a) one-shot (k=0) | 6/23 (26%) | 1/8 | 3/8 | 2/7 |
| (b) feedback ≤ 3 rounds | 12/23 (52%) | 3/8 | 4/8 | 5/7 |

- First-attempt failures (17): collision 8, out-of-bounds 8, floating 1. All 54 failed attempts: collision 25, OOB 16, floating 8, unstable 5; 7 truncated by max_tokens.
- Cost ≈ ¥0.20 total (≈ ¥0.017 per valid sample).
- **Quality warning**: many valid samples are degenerate solid stacks of a single brick type (6 of 12). verify checks physics, not semantics; distilling these directly teaches "stack a block".

## Step 2 (100 training prompts, filter as feedback; thresholds fixed before running)
- Filter: ≥ 3 brick types; single-type share ≤ 70%; fill ratio ≤ 85%; ≥ 10 bricks. A prompt suffix asks to model the described shape (no solid blocks, ≥ 3 brick types); filter failures are returned as "quality" errors in the same ≤ 3-round loop.
- Result: 24/100 valid (5 at the first attempt); small 10/34, medium 10/33, large 4/33. Failures over 371 attempts: collision 149, OOB 109, quality 53, floating 18, unstable 17, checker error 1; 104 truncated.
- Cost ¥2.50 (≈ ¥0.104 per final valid sample, ~6x step 1).
- Un-aligned voxel IoU (STABLE method): mean 0.069; AUROC 0.63 for detecting degenerate structures — not usable as a filter.

## Step 3: aligned IoU and CLIP (thresholds fixed before scoring)
- Aligned IoU (max over integer x/y shifts): degenerate 0.167, train100 0.134, random human pair 0.151. AUROC for detecting degenerate structures 0.43 (worse than chance). IoU measures overlap with one specific target, not whether a structure matches its description.
- CLIP set-up follows BrickGPT §5.2 ("A LEGO model of {caption}", openai/clip-vit-base-patch32, raw cosine similarity).
