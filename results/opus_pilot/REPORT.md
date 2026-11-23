# Opus hand-designed control (10 prompts, no API calls)

**Setup**
- Prompts: the first 10 of the 20 harness_vision training prompts (aligned one-to-one with DeepSeek arms A/B).
- Method: shapes designed from the caption only and generated procedurally with `tools/opus_pilot.py` (define per-layer voxels, then greedily tile each layer with allowed bricks, alternating orientation preference between odd/even layers to stagger seams). The target structure is used only to compute IoU, never for construction.
- Check: `check(use_gurobi=True)`; diversity via `distill_filter.keep`; IoU via `distill_shape.aligned_iou` (same code as harness_vision).
- "Rounds" = number of design revisions driven by checker errors (including the first), max 3.
- Renders: `renders/{id}.png` with axes fixed to 0..20. Output: `samples.jsonl`.

## Per-prompt results

| id | bucket | caption | physics | filter | bricks | rounds | IoU (Opus) | IoU A | IoU B | first-round error / shape note |
|---|---|---|---|---|---|---|---|---|---|---|
| c59ff7bc | small | guitar | ✓ | ✓ | 24 | 1 | 0.097 | 0.064 | 0.056 | Lying flat: body with sound hole, long neck, head. Recognizable. |
| 88ae8881 | medium | small square table | ✓ | ✓ | 35 | 1 | 0.133 | 0.066 | 0.050 | 2x2 legs, apron, two-layer top. |
| bc7d1523 | large | slatted bench | ✓ | ✓ | 45 | 3 | 0.047 | 0.061 | 0.039 | Mid-slats floating; rebuilt with a frame under the slats. |
| 3f6b3d88 | small | train | ✓ | ✓ | 73 | 1 | 0.192 | 0.204 | 0.171 | Hollow car with windows, wheels, stepped roof. |
| ea150399 | medium | boxy small car | ✓ | ✓ | 43 | 2 | 0.456 | 0.243 | 0.200 | Front row unsupported → unstable; wheels moved to the ends. |
| 9252da4b | large | square-based long-neck bottle | ✓ | ✓ | 99 | 2 | 0.278 | 0.167 | 0.299 | Shoulder rested only on inner wall → unstable; solid cap plate. |
| 0e0664ab | small | bus | ✓ | ✓ | 90 | 2 | 0.192 | 0.126 | 0.098 | Roof middle floating; width 8, roof of 8x1 bricks spanning both walls. |
| 9751a363 | medium | stacked tall planter | ✓ | ✓ | 103 | 2 | 0.099 | 0.464 | 0.207 | Rim offset from wall → floating; 2-wide rim. |
| 4c39a999 | large | reclined chair | ✓ | ✓ | 42 | 1 | 0.183 | 0.158 | 0.181 | Backrest shifted back every two layers. |
| b9f8fe18 | small | long layered boat | ✓ | ✓ | 58 | 2 | 0.214 | 0.164 | 0.166 | Hollow hull ring not on the keel; lower hull made solid. |

## Summary (same 10 prompts)

| | Opus (hand) | DeepSeek A | DeepSeek B | random human pair |
|---|---|---|---|---|
| Physics valid | 10/10 | 10/10 | 10/10 | – |
| Diversity filter | **10/10** | 5/10 | 6/10 | – |
| Mean aligned IoU | **0.189** | 0.172 | 0.147 | 0.151 |

## Conclusions
- IoU gains are small and noisy (one prompt, the car at 0.456, lifts the mean by ~0.03). Captions are under-determined; aligned IoU mostly measures whether proportions and orientation happen to match the single target.
- The clear advantage is **sample quality**: 10/10 pass the filter, all Gurobi-stable, all subjectively recognizable.
- Failure modes are concentrated: floating bricks, loads resting only on hollow inner walls, unsupported mid-spans — each fixed in one revision.
