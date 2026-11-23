# Research log (2026-11-17 → 2026-11-23)

Condensed English version of three internal planning documents: the research trajectory, the direction/pre-registration document, and the debug log. Entries are chronological. Pre-registered rules for the final experiments are reproduced in full in [PREREGISTRATION.md](PREREGISTRATION.md). Many of the earlier side-studies below are not part of the paper; their code is still in `brickexp/` (`tiny/`, `screen/`, `truss/`) and their small summaries are in `results/`.

## Overview: idea → experiment → feedback → next step

| # | Dates | Question | Experiment | Result / feedback | Lesson → next |
|---|---|---|---|---|---|
| 0 | ≤ 11/17 | Original proposal: add new brick types, learn physics | — | Course staff: too hard for available compute; suggested bridge-building ICL | Narrow the problem |
| 1 | 11/17–18 | Can a model "grow up": internalize verifier-assisted experience and depend less on the verifier? | C0/C1: Qwen SFT + checker-budget curve; GRPO planned | Qwen SFT 0 → ~10% stable (BrickGPT order of magnitude); not enough compute for GRPO, paused | Need a cheaper, controllable setting |
| 2 | 11/18–19 | Does trial-and-error data help because of the **content of the correction** or just the format / extra tokens? | From-scratch tiny model, 4 arms (S / R+F / R+F̃ / R) × 3 seeds, exact Gurobi feedback | No significant difference among arms ("downgrade"); v1 voided due to a confound | Negative result; write down what a negative result means before running; check literature first |
| 3 | 11/19 | A **conservative physics screener** (GNN + one-sided conformal) that defers to the exact solver when unsure? | Leave-one-category-out over 21 LEGO categories | In-distribution miss ≤ 5.7%, 67–72% solver calls saved; guarantee held for 19/21 held-out categories | Holds on LEGO → test on real engineering |
| 4 | 11/19–20 | Does the LEGO conclusion hold on **real truss FEM**? | Leave-one-family-out over 8–10 truss families, several rounds | Partial transfer (4/8); Howe miss 38%; diagnosed a **shape shortcut** (Pratt shape ⇒ safe); flip augmentation cuts Howe miss 31.8% → 1.5%; mechanism-aware alarm ineffective | The model learned a shape shortcut, not mechanics |
| 5 | 11/18–19 (team member) | Can small models learn to build from in-context examples? | imitate / combine, 20 prompts each | Copy examples verbatim; 0 successes on combination | Examples only provide shortcuts |
| 6 | 11/20 meeting | Pivot to **harness + distillation** (approved by course staff; course requires training) | — | — | New main line |
| 7 | 11/20–21 | Large model builds data inside the harness; SFT a small model | DeepSeek 96 samples → Qwen-1.5B | 14/90 physics-valid but **all solid blocks** (blind recognition 0/14) | Physics metric is gameable → add blind recognition; too little data |
| 8 | 11/21–22 | More data + new categories | Opus builds 187 + 59 new-category; mix A/B | B 26/90 vs A 15/90 (Gurobi); new categories fail | Does the gain come from AI data itself? |
| 9 | 11/22 | Separate "source" from "volume" | C group (same amount of human data), second seed | C 21; B 26 / 20 → gain explained by volume and seed noise | AI data ≈ human data: substitutable, not better |
| 10 | 11/22 | Large-scale Gemini data | 665 paired samples, category test | All physics-valid but boxier (aspect ratio 1.6 vs 2.6) | Teacher bias is inherited; stop experiments |
| 11 | 11/21–23 (team member) | Large-model ICL in an agentic builder; distilling the agent process into a small model | DeepSeek 20 pairs; 275 trajectories SFT Qwen3.5-2B | ICL: faster, not better; distillation: 0/10, loops in place | Multi-step closed-loop behaviour does not transfer |

**Common thread — models learn shortcuts, not physics.** Step 2: correction content does not matter (format only?). Step 4: the GNN learns "truss shape ⇒ safe". Step 5: small models copy examples. Step 7: "solid ⇒ stable". Step 9: gains come from data volume, not from AI data being "easier to learn". Step 11: agent distillation learns repeated actions. The same phenomenon recurs in both LEGO and truss FEM: before AI can learn engineering design, it has to get past shortcut learning, and LEGO plus an exact verifier is a cheap test bed for finding and diagnosing shortcuts.

---

## Phase 1 — Direction v2: content vs. format of trial-and-error feedback (11/18–19)

**Question.** When trial-and-error experience makes a from-scratch model internalize a skill, is it the *content* of the correction or merely the format and extra tokens? Prior work disagrees (Physics of LMs 2.2, arXiv 2408.16293: retry data helps; arXiv 2505.13775: wrong or irrelevant traces help as much as correct ones). LEGO has an exact external checker (Gurobi force balance) that can say which brick fails and why, so feedback content can be manipulated precisely.

**Design.** A 10–50M-parameter decoder-only transformer trained from scratch with a custom vocabulary (brick types, x/y/z coordinates, feedback tokens), conditioned on the category label (21 classes). Success data from StableText2Brick; trial-and-error trajectories generated by perturbing stable structures ("attempt → feedback → correction"). Four token-matched arms: **S** (successes only), **R+F** (retries + true checker feedback), **R+F̃** (retries + shuffled feedback), **R** (retries, no feedback). Pre-registered verdicts: "content matters" (R+F significantly better than R+F̃ and R), "format suffices", or "downgrade to course project" (no differences).

**Revisions before results (all written before the relevant numbers were seen).**
- §4a: floor effect — nearly all failures were geometric bookkeeping (collision, out-of-bounds), not physics. Geometry was handed to a tool (masking at generation time).
- §4b: per-structure success is a product of per-brick correctness, so it sits at the floor; switched to per-brick metrics (floating-brick fraction, position of first error).
- §4c: masking was found to double the floating fraction (it pushes the model to unsupported fallback cells), so masking was replaced by a deterministic **post-hoc geometry filter**. Floating is judged at placement time, with "hanging under a later brick above" allowed; acceptance test: 0/42,738 floating bricks on the training targets.
- §4d: **confound found** — the LM loss included the 26,629 injected bad bricks (51% floating), so retry arms learned to produce errors themselves; arm R also had 7% fewer tokens. v1 was stopped and archived; v2 masks the loss on bad-brick tokens and token-matches R.

**Result (§4e, v2, 12/12 runs).** k=0 success: S 1.35%, RF 0.40%, RF̃ 0.48%, R 0.87%; floating fractions 0.170–0.191, all 3-seed CIs overlapping. **Verdict: downgrade** (no arm differs). The v1 by-product — "put bad bricks in the loss and the model learns to make errors" — is noted as a discussion point only.

## Phase 2 — Conservative physics screener (11/19)

**Pre-registration (§4g).** A small GNN predicts stability from solver labels; one-sided conformal calibration bounds the rate of "judged safe but actually unstable" at α ∈ {1%, 5%}. Measure miss rate and solver calls saved in-distribution and under shift (held-out category, larger size), plus a recalibration curve with N ∈ {0, 10, …, 1000} target labels.

**Results (§4h, §4i).** In-distribution: α=1% → miss 0.77%, 67% saved; α=5% → miss 5.69%, 72% saved; AUC 0.998. Under shift most categories keep the guarantee (bottle, vase: miss 0–4%, 72–95% saved); guitar fails at α=5% (19.6%) as a *high-confidence* error on tall, high-moment structures. A 1.93M-parameter GNN generalised slightly worse than the 50k one. Leave-one-category-out over 20 more categories: only 1 category exceeded 10% miss → verdict **FAILURE_RARE**; a label-free alarm could not be evaluated (too few positives).

## Phase 3 — Real truss FEM replication (11/19–20)

**Pre-registration (§4j).** Self-generated 2D/3D trusses solved by the linear-elastic direct stiffness method (verified against 3 analytical solutions to 1e-9). Unsafe if any member exceeds allowable stress, buckles (Euler), or deflection > span/250. Steel, E = 200 GPa; 8 families (Pratt, Howe, Warren, K, Fink, bowstring, space frame, random topology). Same GNN screener (74k parameters), leave-one-family-out, size shift.

**Results.**
- §4j: **PARTIAL** transfer (4/8 families meet the rule). Howe miss 38%, Fink 18%, space frame 100% (training had only 2D trusses), random 34%; size shift 14%. N=1000 recalibration restores miss ≤ 5.7% everywhere. Geometric alarm AUROC 0.78 (< 0.8, ineffective).
- Diagnosis: the "failure-mechanism change" hypothesis was partly rejected; the better explanation is a **shape shortcut** — the parallel-chord + vertical + diagonal silhouette appears in training only as Pratt, whose diagonals are in tension and almost always safe.
- §4k rounds 2–6 (10 families incl. two new 3D families; 10 epochs, then a 20-epoch replication): round 2 sits on the border between "no transfer" and "partial transfer" (5/10 at 20 epochs); round 3 "deployable" (median N* = 75–100 target labels restores the guarantee); round 5 **supports the shortcut hypothesis** (diagonal-flip augmentation: Howe miss 31.8% → 1.5%; 32.1% → 1.9% at 20 epochs); round 4 mechanism-aware label-free alarm AUROC 0.50 (ineffective).

## Phase 4 — Harness + distillation (paper main line, 11/20–23)

- **DeepSeek pilot** (`results/distill_pilot`): 26% one-shot / 52% with ≤ 3 feedback rounds on 23 prompts; many valid samples were degenerate solid stacks → added a diversity filter. Voxel IoU (raw or aligned) could not separate degenerate from good samples (AUROC 0.43–0.63; random human pairs score 0.151).
- **Vision feedback** (`results/harness_vision`): renders in the feedback loop did not improve shape (pre-registered criterion not met), at ~3x cost.
- **Opus hand-designed control** (`results/opus_pilot`, `data/opus_gen`): 187/200 pass; 10/10 pass the diversity filter on the pilot prompts.
- **Distilled SFT** (`results/distill_sft`): 96 samples → Qwen2.5-1.5B goes 0/90 → 12/90 valid, but blind recognition of the valid outputs is **0/14** vs 12/14 for matched dataset structures (`results/recognition_eval_sft_v2`).
- **Mix A/B/C** (`results/mix_A`, `results/mix_B`, `results/final_small`, `results/paper`): Gurobi-checked k=0 valid on the 90 prompts — base 0, A 15, B 26 and 20 (two seeds), C 21; no detectable difference between AI and size-matched human data; **0/82** valid structures recognized.
- **Gemini paired data** (§4n): physics-valid but boxier (aspect ratio 1.6 vs 2.6 for human structures) — teacher bias is inherited; experiments stopped.

---

## Debug log (condensed)

Rule adopted early: every failure gets *symptom / root cause / evidence / fix*; no job is resubmitted unchanged without a root cause; two consecutive failures of the same kind trigger a platform re-evaluation.

| Job | Symptom | Root cause | Fix |
|---|---|---|---|
| C0 v2 | vLLM fails to start | flashinfer JIT cannot find `-lcuda` | symlink `libcuda.so.1 → libcuda.so` |
| C0 v3 | Gurobi unavailable | API-pushed Kaggle kernels cannot read Secrets | connectivity check on Kaggle; Gurobi re-check locally |
| C0 v4/v5 | vLLM warm-up fails (`cutlass`) | incompatible vLLM build | vllm 0.17.1 + `enforce_eager` |
| C0 v6 | 8/30 false positives | parser read only the first brick per line | parse every brick per line; brick-count mismatch = failure; unit test |
| C1 #1–#5 | KeyError / SyntaxError / TypeError / OOM | old transformers; notebook quoting; removed `warmup_ratio`; full-vocab (248k) logits | isolated transformers 5; `ast.parse` all cells; `warmup_steps`; chunked cross-entropy, single GPU |
| C1 dry-run | loss = grad_accum × expected | Trainer did not divide a mean-reduced custom loss | `model_accepts_loss_kwargs=False` |
| C1 v10 | dtype mismatch in chunked CE | fp32 hidden states vs fp16 lm_head under AMP | cast before matmul; mixed-dtype unit test |
| C0 v8 | vLLM `EngineDeadError` | gated-delta-rule prefill state not in KV budget; 300 concurrent requests | `gpu_memory_utilization=0.75`, `max_num_seqs=64` |
| Gurobi | ~11–17 s per structure vs ~0.35 s reported | ill-conditioned LP (one dense balance-residual row) | `NumericFocus=3` (4.2× faster, 80/80 identical verdicts) |
| C1 v11 | tokenizer class missing at eval | transformers-5 artefacts unreadable by vLLM's transformers | `brickexp/rebase_qwen35.py` copies base repo and swaps in 320 LM tensors; greedy outputs verified identical |
| Tiny (Colab) | lost runs, OOM, stale code, CRLF scripts, idle disconnects, quoting bug in job lists | no KV cache; whole-batch cache; upload/unzip race; Windows line endings; `"${JOBS:-...}"` not word-split | KV cache + `--eval_bs`; version asserts before launch; LF-only scripts; output-based success checks (`check_out.py`) |
| Tiny v1 | confounded comparison | injected bad bricks were in the LM loss | v2 masks bad-brick loss (see Phase 1); runs completed on Modal |
| Screener (Kaggle) | dataset not attached; `gurobipy` missing | Kaggle auto-extracts zips; vendored module imports gurobipy at top level | locate extracted dir; solver-free parsing path; local replay of the kernel in a clean venv before each push |
| Truss | weights not saved; GBK-encoded patch broke `§`; 3D box family was a mechanism; K-truss flip makes mechanisms | — | save weights + scores; always write UTF-8; fix supports (zero-energy-mode count); restrict flip augmentation to Pratt + bowstring |
| Distill | spend cap exceeded slightly | balance checked only at round start | check every N calls |
| Renderer | all renders auto-zoomed | 3D axes autoscaled to filled voxels | fixed axes 0–20 (earlier `harness_vision` renders use the old renderer) |
| Mix A/B (Kaggle) | bootstrap assertion; missing parent dir; "max 2 GPU sessions" | auto-extracted zip; `cp` without `makedirs`; quota | fallback to extracted dir; `makedirs`; queued retries |
| Paper re-check | B_s0 25 vs 26 earlier | one `checker_error` (Gurobi timeout) under CPU saturation | retry `checker_error` up to 3× and regenerate all numbers |
