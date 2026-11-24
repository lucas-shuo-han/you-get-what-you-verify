# In-context learning on BrickBench (team experiments)

Source: teammates' experiment reports for the two in-context settings (imitate and combine). Raw generations are not included in this repository; the numbers below are transcribed from those reports.

Setup (both settings): Qwen2.5-Coder-1.5B-Instruct, Q8_0 quantization, seed 42, greedy decoding, 3072-token generation cap. Models write Python build programs that are checked by the official BrickBench verifier.

## Imitate (20 tasks; two reference programs for the target task in the 2-shot prompt)

| Prompt | Valid | Notes |
|---|---|---|
| 0-shot | 0/20 | |
| 2-shot | 3/20 | All 3 valid outputs are near-verbatim copies of a prompt example (line similarity 0.99–1.0); one is byte-identical. |

A further 9 outputs were near-copies (line similarity 0.95–0.98) that failed with `NameError`.

## Combine (20 pairs; "build X using parts of A and B")

| Prompt | Extractable code | Exported and passed physics | Genuine combinations |
|---|---|---|---|
| 0-shot | 0/20 (all refusals) | 0/20 | 0/20 |
| 2-shot | 15/20 | 3/20 | 0/20 |

All 3 passing 2-shot outputs are byte-identical to one of the prompt examples.
