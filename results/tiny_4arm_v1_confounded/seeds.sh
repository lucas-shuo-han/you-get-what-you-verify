#!/bin/bash
# Seeds 1-2, original recipe. Per job: train (+ unmasked eval at k=0 only, to save time) -> masked eval k<=10.
# Resumable: skips stages whose outputs exist on Drive.
while pgrep -f "bash /content/geom.sh" >/dev/null; do sleep 30; done
cd /content/RLtheLEGO
D=/content/drive/MyDrive/RLtheLEGO_tiny4arm; G=/content/drive/MyDrive/RLtheLEGO_tiny4arm_geom
for j in ${JOBS:-"RF:1 S:1 RFs:1 R:1 RF:2 S:2 RFs:2 R:2"}; do
  arm=${j%%:*}; seed=${j##*:}; n=${arm}_s${seed}
  mkdir -p $D/$n $G/$n
  if [ ! -f $D/$n/eval.jsonl ]; then
    rm -f $D/$n/model.pt
    echo "=== $(date) train $n" | tee -a $D/runner.log
    python -m brickexp.tiny.train --data data/tiny/n4000_s0 --arm $arm --seed $seed --out $D/$n --eval_k 0 >> $D/$n/run.log 2>&1
    echo "=== $(date) end train $n rc=$? $(grep EVAL $D/$n/run.log | tail -1) [unmasked eval k=0 only]" | tee -a $D/runner.log
  fi
  if [ ! -f $G/$n/eval_geom.jsonl ]; then
    echo "=== $(date) start $n" >> $G/runner.log
    python -m brickexp.tiny.train --data data/tiny/n4000_s0 --arm $arm --seed $seed --out $G/$n --eval_model $D/$n/model.pt --eval_only --geom_mask >> $G/$n/run.log 2>&1
    echo "=== $(date) end $n rc=$? $(grep EVAL $G/$n/run.log | tail -1)" >> $G/runner.log
  fi
done
echo SEEDS_DONE | tee -a $D/runner.log
