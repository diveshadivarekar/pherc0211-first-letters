#!/usr/bin/env bash
# Usage: ink_depth_sweep.sh <render.zarr> <out_prefix> <ckpt> <d1> <d2> ...   (66-layer render, mesh at layer 33; 17-layer model window)
set -u
RENDER=$1; PREFIX=$2; CKPT=$3; shift 3
for d in "$@"; do
  c=$((33 + d)); s=$((c - 8)); e=$((c + 9))
  out="${PREFIX}_d${d}.tif"
  [ -f "$out" ] && continue
  D:/divesh-github/inkenv/Scripts/python.exe -m vesuvius.ink_detection.inference.infer "$RENDER" "$CKPT" "$out" \
     --overlap 0.5 --blend-mode hann --batch-size 4 --direction both --layer-start $s --layer-end $e > /dev/null 2>&1
  echo "done d=$d"
done
echo SWEEP_DONE
