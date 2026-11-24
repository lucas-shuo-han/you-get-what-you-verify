#!/bin/bash
# §4c queue v3. Fixes: (a) JOBS default is now split into words (v2 used ${JOBS:-"..."} which made ONE job "RF:1 ... R:2" -> only RF_s2);
# (b) checks that per_brick_metrics.py (imported by train.py eval) is importable before starting.
# v4: success verified by OUTPUT (check_out.py), not log lines.
# Part 1: seeds 1-2 train + FULL unmasked eval k<=10 (raw saved). Part 2: seed-0 unmasked raw re-eval (--eval_tag raw).
cd /content/RLtheLEGO
while pgrep -f "brickexp.tiny.train" >/dev/null; do sleep 20; done  # let in-flight job from seeds3 finish
python -c "import brickexp.tiny.per_brick_metrics, brickexp.tiny.train" || { echo "IMPORT FAIL"; exit 1; }
D=/content/drive/MyDrive/RLtheLEGO_tiny4arm
DEF="RF:1 S:1 RFs:1 R:1 RF:2 S:2 RFs:2 R:2"
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
echo SEEDS_DONE | tee -a $D/runner.log
R=$D/seed0_raw
for arm in RF S RFs R; do n=${arm}_s0; mkdir -p $R/$n
  ls $R/$n/eval_raw*.jsonl >/dev/null 2>&1 && continue
  echo "=== $(date) start raw $n" | tee -a $D/runner.log
  python -m brickexp.tiny.train --data data/tiny/n4000_s0 --arm $arm --seed 0 --out $R/$n --eval_model $D/$n/model.pt --eval_only --eval_tag raw >> $R/$n/run.log 2>&1
  rc=$?
  echo "=== $(date) end raw $n rc=$rc $(grep EVAL $R/$n/run.log | tail -1)" | tee -a $D/runner.log
  [ $rc -ne 0 ] && { echo "FAIL raw $n rc=$rc" | tee -a $D/runner.log; exit 1; }
  python /content/check_out.py $R/$n eval_raw.jsonl | tee -a $D/runner.log; [ ${PIPESTATUS[0]} -ne 0 ] && exit 1
done
echo ALL_DONE | tee -a $D/runner.log
