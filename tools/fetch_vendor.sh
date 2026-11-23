#!/usr/bin/env bash
# Fetch third-party code used by brickexp at the exact commits the experiments ran with.
# BrickGPT (MIT): brick library + stability analysis used by brickexp/checker.py.
# STABLE: only needed for tools/distill_iou.py (raw voxel-IoU baseline).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor
if [ ! -d vendor/BrickGPT ]; then
  git clone https://github.com/AvaLovelace1/BrickGPT.git vendor/BrickGPT
  git -C vendor/BrickGPT checkout 9c53c7cb65f9de0df89562cc74b2d2e270fbcab4
fi
if [ "${WITH_STABLE:-0}" = "1" ] && [ ! -d vendor/STABLE ]; then
  git clone https://github.com/miniHuiHui/STABLE.git vendor/STABLE
  git -C vendor/STABLE checkout a42b0c810b62e6e269829b7e52b6d90d19214de1
fi
echo "vendor ready"
