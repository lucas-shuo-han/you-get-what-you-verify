#!/usr/bin/env bash
# kaggle CLI wrapper: loads a raw token from ~/.kaggle/kaggle.json into env (never echoed)
f="$HOME/.kaggle/kaggle.json"
if [ -s "$f" ] && [ "$(head -c1 "$f")" != "{" ]; then export KAGGLE_API_TOKEN="$(tr -d '\r\n ' < "$f")"; fi
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
exec kaggle "$@"
