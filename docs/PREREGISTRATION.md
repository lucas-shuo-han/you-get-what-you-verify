# Pre-registered decision rules (§4l, §4m, §4n)

English translation of the three pre-registrations that govern the experiments reported in the paper. They were written into the project's planning document *before* the corresponding results were downloaded or inspected; the text below is a faithful translation of the originals (section numbers kept). Dates are 2026.

---

## §4l. Mixed-data A/B experiment (pre-registered 2026-11-22; at this point neither the mix-a nor the mix-b results had been downloaded and no output had been seen)

- **Groups**: A = SFT on 4000 human examples; B = 4000 human + about 3x AI-generated data (including new categories); base = untrained model (floor).
- **Evaluation**: 28 new-category prompts and the original 90 prompts, both at k=0 (one generation). Physics validity is always decided by a **local Gurobi re-check**. Recognition uses blind judging (axes fixed to 0–20, images shuffled, judge blind to group, candidate-class protocol as in `recognition_eval_sft_v2`).
- **Primary metric (new categories)**: number of prompts that are "valid and recognized" / 28. The denominator is fixed at 28; a prompt that is not valid counts as "not recognized", so the denominator does not change when only valid samples are inspected. Top-1 recognition is primary; top-2 is reference only.
- **Secondary metrics**: physics-valid prompts / 28; number passing the diversity filter (≥ 3 brick types, not a single solid block).

**Decision rules (28 new-category prompts)**
- **B helps**: primary metric B − A ≥ 5 prompts **and** one-sided Fisher exact test p < 0.05. Both are required.
- **Helps physics only**: physics-valid B − A ≥ 5 prompts and p < 0.05, but the primary metric does not meet the rule above. Phrase as "AI data increased validity but not recognizability", consistent with the decoupling of physics validity and semantics seen in the distillation experiment.
- **No difference**: neither holds. Only "no difference detected at 28 prompts" may be written, not "AI data is useless"; 28 prompts can reliably detect only differences of about 5 prompts or more.
- **B worse**: A − B ≥ 5 prompts and p < 0.05; report as is.

**"No harm" on the original 90 prompts**
- Valid count B ≥ A − 5 **and** two-sided Fisher test not significant (p ≥ 0.05). Otherwise write "mixing in AI data hurt in-distribution performance".

**Base floor**
- If a group's primary metric on the new categories is ≤ base + 2 prompts, that group is judged "learned nothing on new categories"; the A/B comparison is then descriptive only.
- If both A and B are at the floor, the overall verdict is "new categories not learnable at this scale".

**Other constraints**
- The recognition judge is a single model (Claude) and the samples are small; this must be stated in the report.
- After results arrive, the primary metric, thresholds and prompt set may not be changed. If the Gurobi re-check and the Kaggle connectivity check disagree, Gurobi is authoritative and both are reported.
- Only one seed, so every conclusion is labelled "preliminary".

---

## §4m. Final-experiment pre-registration (evening of 2026-11-22, written before any new result)

Purpose: separate "data source" from "data volume". Three groups with the same size and the same number of steps (STEPS = ceil(4852/16) = 304; batch 2 x accumulation 8; lr 2e-4; LoRA r16; Qwen2.5-1.5B-Instruct; Kaggle T4):
- A = 4000 human; B = A + 284 AI x3 (4852); C = A + 284 extra human x3 (4852; drawn with Random(2027), disjoint from A and from the evaluation set).
- Seeds 0/1/2 (`train_sft --seed`); A_s0 and B_s0 reuse the already completed runs (same hardware, same hyperparameters).
- Training-length check: A and B once each with 3x steps (912 steps), seed 0.

Primary metric: number of the original 90 prompts that pass the **local Gurobi re-check** at k=0. Secondary metrics: blind top-1 recognition (valid and recognized); the 28 new-category prompts.

Decision (on the mean over 3 seeds):
- "AI source helps": mean(B) − mean(C) ≥ 5 prompts **and** one-sided Fisher p < 0.05 on the 3 seeds pooled (B 270 prompts vs C 270 prompts).
- "Just data volume": mean(C) − mean(A) ≥ 5 and |mean(B) − mean(C)| < 5.
- Training length: if B − A is still ≥ 5 at 3x steps, write "the advantage persists with longer training"; otherwise "the advantage exists only in the low-compute regime".
- Any other outcome: "no reliable difference". Metrics and thresholds are not changed after results arrive. C's extra human data contains no new categories, so new-category results are descriptive only.

---

## §4n. Final experiment (supersedes the data setup of §4m; written on the evening of 2026-11-22, before any F* result)

The team decided to use newly generated Gemini data instead. Paired design:
- FA = 4000 human (same as A); FB = FA + 665 Gemini structures (physics_ok and filter_ok, one per structure_id, no upsampling);
  FC = FA + **the human answers for the same 665 structure_ids** (same prompts; the only difference is the answer source). Extra answer characters: Gemini 1.08M, human 1.12M (same order of magnitude).
- All three groups use 292 steps (= ceil(4665/16)); other hyperparameters as in §4m; seeds 0/1/2; training-length check: FA and FB at 3x steps (876), seed 0.
- Primary metric, decision thresholds and the "preliminary" label are exactly as in §4m (with A/B/C replaced by FA/FB/FC). The §4m runs already in progress, C_s0 and B_s1 (284 older AI samples), serve only as small appendix data points.

---

### Outcome notes (added for the public release, not part of the pre-registration)
- §4l: all groups were within base + 2 on the new-category primary metric → "new categories not learnable at this scale" (see `results/paper/REPORT.md`).
- §4m: with the seeds that completed (B: 26 and 20; C: 21; A: 15), no comparison met its threshold → "no reliable difference" between AI and human data of equal size.
- §4n: the Gemini data was found to inherit a teacher shape bias (boxier structures) and the F* runs were stopped before completion; the paper reports §4m only. Notebooks for F* runs are kept in `notebooks/final2/` for completeness.
