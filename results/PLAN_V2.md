# V2 plan (CS182 LEGO project) — draft for review

## Goal
Turn V1's "small model learned stability but not semantics (0/82 recognized)" from a possibly-undertrained failure into a diagnosed phenomenon: (a) separate *selection bias* (physics filter discards semantic attempts) from *underfitting* (not enough data/steps/capacity), and (b) show where, if anywhere, recognition rises above zero as the training budget grows. Deadlines: V2 Dec 7, final + repo + checkpoints Dec 12.

## Experiments (in order)
| ID | What | Compute | Decides |
|---|---|---|---|
| E0a | Render and blind-judge students' outputs that failed only by brick collision (A/B/C, ~60–90 structures) | none (API judge) | selection bias vs not |
| E0b | Geometry statistics: brick count, bounding-box aspect, solidity (filled fraction) for human ground truth vs student-valid vs student-collision-failed | none | whether the physics filter keeps only simple solids |
| E0c | Second judge from another model family (Gemini) on all V1 recognition sheets; two team members blind-label 40–50 images; report agreement (Cohen's κ) | none | judge bias |
| E1 | Group A (human data) trained 5 epochs on Kaggle T4, evaluated at every epoch (running now) | free Kaggle | does loss / validity / recognition keep improving with steps |
| E2 | Budget scan on Modal (A100/H100): data size {4k, 16k, 47k} × {LoRA r16, full fine-tune}, up to 3 epochs, evaluated every 0.5 epoch. Start with one small cell to measure real throughput before launching the grid | ~6–10 GPU-h | where recognition leaves zero; "stability before semantics" curve |
| E3 | Re-run A/B/C with full tracking (below), 3 seeds, fill C seed-1 new-category eval | ~2–3 GPU-h | V1 promises: validation curves, seeds |
| E4 | Repository: one-command solver-free reproduction, release LoRA checkpoints and per-run trajectories | none | reviewer reproducibility |

Out of scope for V2: RL/GRPO, dual-verifier training, new frontier-model data generation, agent-trajectory distillation.

## Training-trajectory tracking (revised after GPT/Gemini review, results/plan_v2_review/)
Every run produces the following; a run without them does not count.

1. **Live trajectory** `<run>/trajectory.jsonl`, one event per line (`train` every 10 optimizer steps: loss, LR, grad norm, epoch; `validation` every 40 steps on the fixed held-out set; `snapshot`; `resume`; `end`), each with run_id, session_id, step, wall time, flushed and fsynced. Persistence: local disk; on Kaggle the files reach storage only when a session ends normally (time guard leaves >=15 min); a hard kill loses that session. Not crash-proof beyond that.
2. **Loss definition** (fixed for all runs, written into the manifest): mean NLL over assistant (brick) tokens per micro-batch; prompt and padding masked; rows longer than the length cap are dropped, not truncated (211/4000 for group A, indices in the manifest); micro-batches weighted equally across gradient accumulation (kept as in V1 so E1/E3 stay comparable); validation loss = mean of per-batch losses over the held-out set.
3. **Aligned milestones**: at step 0, at every generation milestone and at the final step the run records a validation loss and keeps an adapter snapshot at that exact step (`traj/step_N`, checkpoint_id `run/step_N`); periodic snapshots every 80 steps for LoRA, milestones only for full fine-tuning.
4. **Generation checkpoints** at step 0, every 0.5 epoch (every epoch for E1) and final: k=0 on the fixed 90 + 28 prompts, vLLM temperature 0.6, seed 0 (as V1); raw outputs kept before any physics filter; offline connectivity/Gurobi re-check and blind recognition keyed by prompt_id and checkpoint_id. Report physical validity, unfiltered semantic success and joint success with explicit denominators; re-sample with 2-3 seeds only where recognition first rises above zero.
5. **Resume**: full HF Trainer checkpoint (optimizer, scheduler, RNG, scaler) every 40 steps, latest kept; on resume a `resume` event with resume_from_step; the plotter discards earlier lines beyond that step (rolled-back branch). Raw logs are never deleted.
6. **Manifest** `<run>/manifest.json` (automatic): args, TrainingArguments, LoRA config, base model, git commit per session, data and eval hashes, examples kept/dropped, effective batch, planned steps, loss definition, library versions, hardware.
7. **Plot + check**: `tools/plot_trajectory.py <run>` builds the canonical trajectory, writes `trajectory.png` (loss panel; validity and semantic/joint panels when evaluations exist) and `trajectory_check.json` (duplicates, missing step-0, milestone alignment, periodic gaps).
8. **Health**: stop only on NaN/Inf or unrecoverable checkpoint errors; rising validation loss is a warning, never an automatic stop (planned budgets run to completion); two failed sessions in a row -> debug before resubmitting.
9. **Release**: trajectories, manifests, plots, generations and judge outputs to the repo; adapters to a release.

E1 is a transitional run: its first session logged train loss every 10 steps and validation once per epoch; later sessions use the new events. Missing early points are reported as missing, not filled in.
Implemented in `brickexp/train_sft.py` and `tools/plot_trajectory.py`; CPU-tested with a stop/resume across an off-grid milestone and a simulated rolled-back branch.

Added to E0b: compare the 211 length-dropped rows with the kept ones (length, brick count, category).

## Decision rules (stated before seeing results)
- E0a: recognition of collision-failed outputs ≥ 15% (and above the valid-output rate by ≥ 10 points) → selection bias is a main cause; ≈ 0% → not.
- E1/E2: recognition > 10% top-1 at any budget point → "breaks zero"; report the budget at which it happens, with loss and validity at that point.
- E1: run all 5 epochs. If recognition stays 0 throughout while validation loss has flattened or risen, conclude only that more steps within this budget brought no semantic gain, and move budget to data size.
