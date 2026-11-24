#!/bin/bash
# masked re-eval into separate dir; code in /content/geom (does not touch running queue's code)
cd /content/geom
D=/content/drive/MyDrive/RLtheLEGO_tiny4arm; G=/content/drive/MyDrive/RLtheLEGO_tiny4arm_geom; mkdir -p $G
for j in ${JOBS:-"RF:0 S:0"}; do
  arm=${j%%:*}; seed=${j##*:}; n=${arm}_s${seed}
  [ -f $G/$n/eval_geom.jsonl ] && { echo skip $n; continue; }
  [ -f $D/$n/model.pt ] || { echo "no model $n"; continue; }
  mkdir -p $G/$n; echo "=== $(date) start $n" >> $G/runner.log
  python -m brickexp.tiny.train --data data/tiny/n4000_s0 --arm $arm --seed $seed --out $G/$n --eval_model $D/$n/model.pt --eval_only --geom_mask >> $G/$n/run.log 2>&1
  echo "=== $(date) end $n rc=$? $(grep EVAL $G/$n/run.log | tail -1)" >> $G/runner.log
done
echo DONE >> $G/runner.log
