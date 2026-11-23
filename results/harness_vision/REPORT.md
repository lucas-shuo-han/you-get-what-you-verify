# Vision-feedback harness pilot (metrics fixed before running)

**Arms**: A returns only the physics error text. B returns the error text plus two rendered views and asks the model whether the build looks like the caption and what does not match.

**Common setup**
- Model deepseek-flash (image input supported), reasoning_effort=low, max_tokens 16000, temperature 0.6.
- 20 training-split prompts (seed 4242, small/medium/large interleaved), asserted disjoint from the 90 evaluation prompts; same prompts for both arms.
- Up to 3 feedback rounds (≤ 4 attempts). Context window 1: each round carries the initial prompt, the previous output and its feedback.

**Stopping**: A stops when the physics check passes; B stops when the physics check passes **and** the model replies DONE. Final structure = last physics-passing attempt (else the last attempt).

**Metrics**: (1) validity; (2) aligned voxel IoU vs target (valid only; random human-pair reference 0.151); (3) diversity-filter pass rate; (4) cost (cap ¥4).
**Pre-registered decision**: vision feedback "helps" iff B's aligned IoU exceeds A's by ≥ 0.05 and B's validity is no more than 10 pp below A's.

## Results (20 prompts per arm)

| Metric | A: text errors | B: errors + renders |
|---|---|---|
| Physics valid | 20/20 | 20/20 |
| Diversity filter pass | 9/20 | 12/20 |
| Aligned IoU (valid mean) | 0.155 | 0.140 |
| Mean attempts | 1.5 | 3.55 |
| API calls | 30 | 71 |
| Output tokens | 138k | 420k |
| Truncated | 1 | 5 |
| Estimated cost | ≈ ¥0.48 | ≈ ¥1.47 |

**Verdict (pre-registered): vision feedback did not help.** B's IoU is 0.015 lower than A's; both are at the random reference (0.151).

**Other observations**: deepseek-flash is highly physics-capable (A: 14/20 valid at the first attempt, 20/20 within 3 rounds). Only 7/20 B episodes ended with DONE; seeing renders led to more edits (diversity 12 vs 9) but not closer shapes, at ~3x the cost. Note: these renders were made before the fixed-axis renderer fix (see `docs/RESEARCH_LOG.md`), which may partly explain the null result.
