# Data

| Path | Contents | Size |
|---|---|---|
| `opus_gen/` | Teacher structures designed by Claude Opus with a support-aware tiler (`batch_*/`, `newcat_*/`): `samples.jsonl`, generator scripts, renders. `opus_all_ok.jsonl` = the 187 physics-valid samples used as AI data in mix B. `newcat_{train,test}.jsonl` = new-category split. | ~13 MB |
| `distill/` | DeepSeek distillation data (`sft_distill.jsonl`); the 96/65-sample subsets are selected by id lists in `results/distill_data_audit/keep_ids.json`. | small |
| `gemini_gen/final/sNNN/samples.jsonl` | Gemini-generated structures for the §4n paired design (`physics_ok`, `filter_ok` flags). | ~30 KB–100 KB each |
| `mix/heldout200.jsonl` | 200 held-out training-split examples for validation loss (seed 4242). | 0.5 MB |
| `mix/newcat_test_eval.jsonl` | the 28 new-category evaluation prompts. | small |
| `mix/manifest.json`, `mix/final_manifest.json` | group sizes and seeds of the A/B/C and FA/FB/FC mixes. | small |

Not included (rebuild them yourself):

```bash
python tools/fetch_stabletext2brick.py     # StableText2Brick -> data/hf_cache (Hugging Face: AvaLovelace/StableText2Brick)
python tools/build_mix_data.py             # data/mix/A.jsonl (4000 human), B.jsonl (A + AI x3), C.jsonl (A + extra human x3)
python tools/build_final_data.py           # data/mix/FA.jsonl, FB.jsonl, FC.jsonl
```

The `self_eval` field in teacher samples held free-text design notes written in Chinese. No analysis uses it, and it has been emptied in this release.
