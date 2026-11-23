# Distilled SFT + 2x2 evaluation (Colab, free T4)

## Setup
- **Base model**: Qwen2.5-1.5B-Instruct.
- **Distillation data**: **only 96 samples** from deepseek-flash (text-only error feedback), on training-split prompts, after the physics check and the diversity filter.
- **Training**: LoRA rank 16, 2 epochs, 12 steps; last logged loss 0.227 (step 10).
- **Evaluation**: the same 90 prompts as C1 and Ref (30 each small/medium/large), up to 3 feedback rounds.
- **Criterion**: connectivity check on Colab (as for C1 on Kaggle, where the Gurobi self-test failed); every accepted SFT structure was then re-checked locally with Gurobi. The base model scored 0 under connectivity, so it needed no re-check.

## 2x2 table (n=90)

| Model | One-shot (k=0) | Feedback ≤ 3 rounds | Criterion |
|---|---|---|---|
| Base Qwen2.5-1.5B | 0/90 | 0/90 | connectivity |
| Distilled SFT (96 samples) | 12/90 (13%) | 14/90 (16%) | connectivity; unchanged after Gurobi re-check (14/14 stable) |
| Reference: C1, Qwen3.5-2B SFT on 4000 human | 10/90 (Gurobi re-check) | not run | connectivity + local Gurobi |
| Reference: BrickGPT (Ref) | 20/90 | — | — |

- By size bucket (k=0 / ≤3 rounds): small 5/6, medium 2/2, large 5/6.
- **Failure causes**: SFT — out-of-bounds 258, collision 45, floating 1, unstable 4. Base — mostly out-of-bounds (338).

## Interpretation
- A 1.5B model trained on just 96 distilled samples matches C1 (4000 human samples) in one-shot validity (13% vs 11%); base models and data sources differ, so the comparison is rough.
- Feedback adds little: +2 prompts.
- **Degeneration**: when the SFT model fails out-of-bounds, its median output length is 265 bricks (distillation-data median: 28) — it emits long sequences until it runs off the grid.
- **None** of the 14 valid structures passes the diversity filter used during distillation.

## Blind recognition (see `results/recognition_eval_sft_v2/REPORT.md`)
- The 14 valid structures: top-1 **0/14**, top-2 0/14. Same-category human structures: top-1 12/14, top-2 13/14.
- The distilled model's "valid" structures are meaningless simple shapes (pillars, slabs, blocks): higher physics validity did not translate into matching the description.

## Files
- `runs/eval_{base,sft}.jsonl`: per-prompt trajectories; `runs/eval_sft.gurobi.jsonl`: local Gurobi re-check; `runs/summary_*.json`, `TABLE_2x2.md`, `sft/train_log.json`.
