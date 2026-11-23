#!/bin/bash
# usage: push_retry.sh LOG args...   retries kaggle_push every 10 min while GPU session limit is hit (max 12h)
log=$1; shift
for i in $(seq 1 72); do
  PYTHONUTF8=1 python tools/kaggle_push.py "$@" > "$log" 2>&1
  grep -q "session count" "$log" || { echo "pushed at try $i"; exit 0; }
  sleep 600
done
echo "gave up"; exit 1
