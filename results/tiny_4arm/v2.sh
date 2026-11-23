#!/bin/bash
# DIRECTION §4d v2 queue. Bad-brick tokens excluded from loss (train.py --loss_mask bad, default); R token-matched.
# Out: MyDrive/RLtheLEGO_tiny4arm_v2. S_s0/S_s1 checkpoints reused from v1 (S path bit-identical) -> eval_only.
# Every job: unmasked eval k<=10 with raw, then check_out.py (output-verified + post_filter per-brick JSON).
cd /content/RLtheLEGO
python -c "import brickexp.tiny.per_brick_metrics, brickexp.tiny.train, brickexp.tiny.pad_r" || { echo "IMPORT FAIL"; exit 1; }
grep -q loss_mask brickexp/tiny/train.py || { echo "NOT v2 train.py"; exit 1; }
python -c "import json;assert 'r_pad' in json.load(open('data/tiny/n4000_s0/stats.json'))" || { echo "NOT v2 data"; exit 1; }
D=/content/drive/MyDrive/RLtheLEGO_tiny4arm_v2; V1=/content/drive/MyDrive/RLtheLEGO_tiny4arm_v1_confounded
mkdir -p $D
for s in 0 1; do mkdir -p $D/S_s$s
  [ -f $D/S_s$s/model.pt ] || { cp $V1/S_s$s/model.pt $V1/S_s$s/train_log.json $D/S_s$s/ && echo "copied v1 S_s$s checkpoint" | tee -a $D/runner.log; }
done
DEF="RF:0 RFs:0 R:0 S:0 RF:1 RFs:1 R:1 S:1 RF:2 RFs:2 R:2 S:2"
JL=${JOBS:-$DEF}
for j in $JL; do
  arm=${j%%:*}; seed=${j##*:}; n=${arm}_s${seed}
  mkdir -p $D/$n
  if [ -f $D/$n/eval.jsonl ]; then python /content/check_out.py $D/$n eval.jsonl | tee -a $D/runner.log; [ ${PIPESTATUS[0]} -ne 0 ] && exit 1; continue; fi
  mode=""; [ -f $D/$n/train_log.json ] && [ -f $D/$n/model.pt ] && mode="--eval_only"
  echo "=== $(date) start $n $mode" | tee -a $D/runner.log
  python -m brickexp.tiny.train --data data/tiny/n4000_s0 --arm $arm --seed $seed --out $D/$n $mode >> $D/$n/run.log 2>&1
  rc=$?
  echo "=== $(date) end $n rc=$rc $(grep EVAL $D/$n/run.log | tail -1)" | tee -a $D/runner.log
  [ $rc -ne 0 ] && { tail -5 $D/$n/run.log; echo "FAIL $n rc=$rc" | tee -a $D/runner.log; exit 1; }
  python /content/check_out.py $D/$n eval.jsonl | tee -a $D/runner.log; [ ${PIPESTATUS[0]} -ne 0 ] && exit 1
done
echo V2_DONE | tee -a $D/runner.log
