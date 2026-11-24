#!/bin/bash
# §4c: seeds 1-2, original recipe; FULL unmasked eval k<=10 (primary). Masked eval only if SEEDS_GEOM=1.
# Resumable: model.pt present but eval missing -> --eval_only.
while pgrep -f "bash /content/geom.sh" >/dev/null; do sleep 30; done
cd /content/RLtheLEGO
D=/content/drive/MyDrive/RLtheLEGO_tiny4arm; G=/content/drive/MyDrive/RLtheLEGO_tiny4arm_geom
for j in ${JOBS:-"RF:1 S:1 RFs:1 R:1 RF:2 S:2 RFs:2 R:2"}; do
  arm=${j%%:*}; seed=${j##*:}; n=${arm}_s${seed}
  mkdir -p $D/$n
  if [ ! -f $D/$n/eval.jsonl ]; then
    mode=""; [ -f $D/$n/train_log.json ] && [ -f $D/$n/model.pt ] && mode="--eval_only"
    echo "=== $(date) start $n $mode" | tee -a $D/runner.log
    python -m brickexp.tiny.train --data data/tiny/n4000_s0 --arm $arm --seed $seed --out $D/$n $mode >> $D/$n/run.log 2>&1
    echo "=== $(date) end $n rc=$? $(grep EVAL $D/$n/run.log | tail -1)" | tee -a $D/runner.log
  fi
  if [ "$SEEDS_GEOM" = 1 ] && [ ! -f $G/$n/eval_geom.jsonl ]; then
    mkdir -p $G/$n
    python -m brickexp.tiny.train --data data/tiny/n4000_s0 --arm $arm --seed $seed --out $G/$n --eval_model $D/$n/model.pt --eval_only --geom_mask >> $G/$n/run.log 2>&1
  fi
done
echo SEEDS_DONE | tee -a $D/runner.log
