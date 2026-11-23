# Recognition-accuracy evaluation (replacing shape IoU)

Judge: **Claude (Opus), viewing images directly**; no external API was called.
Rendering: `tools/opus_pilot.py::render_png`, axes fixed to 0–20 with equal aspect, two views per image (azimuth −60° and 120°). Script: `build.py`.

## Blind protocol
1. `build.py` renders 90 images per group into `renders/{group}/`, shuffles them and copies them to `blind/img_XXXXXX.png` (random ids). The mapping is written to `mapping_DO_NOT_READ.json`.
2. Blind images are tiled 6 per sheet into `blind_sheets/sheet_NN.png`, each tile labelled only with its blind id. The judge saw only these sheets — never file names, group names or the mapping.
3. For each image the judge picks top-1 and top-2 among 21 classes, written to `judgments.jsonl`. Only after all judgments were written was the mapping read for scoring; per-item results are in `scored.json`.

## Results

| Group | n | top-1 | top-2 |
|---|---|---|---|
| human (20 random training structures, upper reference) | 20 | 15/20 = **75%** | 19/20 = 95% |
| ds_A (harness_vision arm A final structures) | 20 | 14/20 = **70%** | 16/20 = 80% |
| ds_B (harness_vision arm B final structures) | 20 | 14/20 = **70%** | 18/20 = 90% |
| ds_distill (20 random distillation samples) | 20 | 12/20 = **60%** | 17/20 = 85% |
| opus (10 opus_pilot structures) | 10 | 9/10 = **90%** | 9/10 = 90% |

## Most frequent confusions (true → judged)
- table → bench (2): low, long slabs with legs; the two classes are nearly indistinguishable.
- camera → car (2): a solid box with a bump.
- pot → basket (2): open hollow boxes.
- vessel → basket / pot / table / bed (4 in distill, 2 in human): vessel is the hardest class.
- train → car / vessel / piano (one error per group, including opus).

## Caveats
- **Judge-affinity risk**: the opus structures were designed by the same model family as the judge with "recognizable at a glance" in mind; the opus 90% is likely inflated. Blinding prevents label peeking, not shared prototypes. Needs human or independent-model re-judging.
- **Small samples**: 20 per group (10 for opus); one image ≈ 5 percentage points; group differences are within noise.
- **Different class mixes**: human is a random sample (7 tables); ds_A, ds_B and opus share the same harness prompts with a more balanced mix that includes hard classes (camera, pot).
- Main takeaway: with fixed axes, ds_A/ds_B top-1 (70%) is close to human structures (75%); distill is slightly lower (60%).
