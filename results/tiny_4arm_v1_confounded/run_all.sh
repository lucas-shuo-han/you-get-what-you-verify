#!/bin/bash
# Sequential 4-arm x 3-seed runner; outputs written straight to Drive; resumable:
#  - eval.jsonl present -> skip; model.pt present (eval missing) -> --eval_only; else train+eval.
cd /content/RLtheLEGO
D=/content/drive/MyDrive/RLtheLEGO_tiny4arm
mkdir -p $D
JOBS=${JOBS:-"RF:0 S:0 RFs:0 R:0 RF:1 S:1 RFs:1 R:1 RF:2 S:2 RFs:2 R:2"}
for j in $JOBS; do
  arm=${j%%:*}; seed=${j##*:}; name=${arm}_s${seed}; O=$D/$name
  if [ -f $O/eval.jsonl ]; then echo "skip $name"; continue; fi
  mode=""; [ -f $O/model.pt ] && mode="--eval_only"
  mkdir -p $O
  echo "=== $(date) start $name $mode" | tee -a $D/runner.log
  python -m brickexp.tiny.train --data data/tiny/n4000_s0 --arm $arm --seed $seed --out $O $mode $EXTRA >> $O/run.log 2>&1
  rc=$?
  echo "=== $(date) end $name rc=$rc $(grep EVAL $O/run.log | tail -1)" | tee -a $D/runner.log
  if [ $rc -ne 0 ]; then tail -30 $O/run.log; exit 1; fi
done
echo ALLDONE | tee -a $D/runner.log
