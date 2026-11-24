#!/bin/bash
cd /content/pb; cp /content/per_brick_metrics.py brickexp/tiny/per_brick_metrics.py
grep -c incremental brickexp/tiny/per_brick_metrics.py
G=/content/drive/MyDrive/RLtheLEGO_tiny4arm_geom; U=/content/drive/MyDrive/RLtheLEGO_tiny4arm
GD=$(for d in $G/*_s*; do [ -f $d/eval_geom.jsonl ] && echo $d; done)
UD=$(for d in $U/*_s*; do [ -f $d/eval.jsonl ] && echo $d/eval.jsonl; done)
python -m brickexp.tiny.per_brick_metrics $GD --k 0 --boot 10000 --out $G/per_brick_k0.json
python -m brickexp.tiny.per_brick_metrics $UD --k 0 --boot 10000 --out $U/per_brick_k0_unmasked.json
echo PB_DONE
