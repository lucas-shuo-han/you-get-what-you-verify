# Experiment index

Each row is one experiment, with where its code, inputs and outputs live. "Paper" marks experiments reported in `paper/main.tex`. Section numbers (§4x) refer to [PREREGISTRATION.md](PREREGISTRATION.md) and [RESEARCH_LOG.md](RESEARCH_LOG.md).

## Paper experiments

| Experiment | Paper | Code | Inputs | Outputs / report |
|---|---|---|---|---|
| DeepSeek harness pilot (text feedback, diversity filter) | §3 teacher | `tools/distill_pilot.py`, `tools/distill_filter.py`, `tools/distill_shape.py`, `tools/distill_iou.py` | StableText2Brick prompts | `results/distill_pilot/` (REPORT.md, p20/train100 jsonl, renders) |
| Vision-feedback harness (arm A vs B) | §3 teacher | `tools/harness_vision.py` | 20 training prompts | `results/harness_vision/REPORT.md`, `summary.json`, renders |
| Opus hand-designed teacher | §3 teacher, Fig. 3 | `tools/opus_pilot.py`, `data/opus_gen/*/gen.py` | captions | `results/opus_pilot/`, `data/opus_gen/` (187 valid + new categories) |
| Blind recognition of teachers vs dataset | Fig. 2 | `results/recognition_eval/build.py` | renders | `results/recognition_eval/` (judgments, mapping, scored.json) |
| 96-sample distillation SFT, 2×2 eval | §4 distill, Fig. 4 | `tools/distill_to_sft.py`, `brickexp/train_sft.py`, `brickexp/eval_2x2.py`, `notebooks/colab_distill_sft.ipynb` | `data/distill/` | `results/distill_sft/` |
| Blind recognition of distilled valid outputs (0/14) | Fig. 2, Fig. 6 | `results/recognition_eval_sft_v2/build.py` | `results/distill_sft/runs/` | `results/recognition_eval_sft_v2/` (v1 in `recognition_eval_sft/` is void) |
| Mix A / B (seed 0) + base | §4, Fig. 4, Fig. 5 | `tools/build_mix_data.py`, `tools/kaggle_mix_push.py`, `notebooks/kaggle_mix_{A,B}.ipynb` | `data/mix/*.jsonl` (rebuilt) | `results/mix_A/`, `results/mix_B/` (eval jsonl, summaries, train logs) |
| B seed 1, C seed 0 (size-matched control, §4m) | §4, Fig. 4, Fig. 5 | `tools/kaggle_final_push.py`, `notebooks/final/*.ipynb`, `notebooks/colab_mix_C_s1.ipynb` | same | `results/final_small/{b1x1,c0x1}/` |
| Gurobi re-check of all student runs | Fig. 4, tables | `results/paper/gurobi_all.py` (earlier version: `results/mix_gurobi_recheck.py`) | eval jsonl | `results/paper/gurobi_all.json` (`results/mix_gurobi_recheck.json` is the superseded first pass) |
| Statistics (Wilson CIs, Fisher tests) + training curves | Appendix, Fig. 5 | `results/paper/curves_stats.py` | `gurobi_all.json`, train logs | `results/paper/stats_table.md`, `stats.json`, `train_curves.png` |
| Blind recognition of A/B/C valid outputs (0/82, §4l) | §5 meaning | `results/paper/recog/build.py` | eval jsonl | `results/paper/recog/` (`summary.json`, per-pool judgments and scores) |
| Gemini paired data (§4n, FA/FB/FC) | discussion only | `tools/build_final_data.py`, `notebooks/final2/` | `data/gemini_gen/final/*/samples.jsonl` | stopped before completion; see RESEARCH_LOG |

## Side studies (not in the paper)

| Experiment | Code | Outputs |
|---|---|---|
| Content vs. format of retry feedback, from-scratch tiny model (§4–§4e) | `brickexp/tiny/`, `tools/build_tiny_notebook.py`, `tools/modal_tiny_v2.py` | `results/tiny_4arm/` (summaries; per-sample evals omitted, >5 MB) |
| Conformal GNN physics screener, LEGO (§4g–§4i) | `brickexp/screen/`, `tools/kaggle_screen.py` | `results/screen_loco/`, `results/screen_diag/` |
| Truss FEM replication (§4j–§4k) | `brickexp/truss/` | `results/truss/` (8-family run); 10-family runs omitted (>5 MB) |
| Early C0/C1 Qwen3.5-2B SFT + checker-budget curve | `brickexp/evaluate.py`, `brickexp/rebase_qwen35.py`, `tools/build_notebook.py`, `notebooks/kaggle_c0_budget_curve.ipynb` | omitted (superseded); described in RESEARCH_LOG |
| Summary figures of the side studies | — | `results/figs/` |
