#!/usr/bin/env bash
# Usage: [CENTER=33] run_reader_v2.sh <render.zarr> <out_prefix> <d1> <d2> ...   (CENTER = layer of the mesh surface: 33 for
# the 66-layer renders in this repo, 15 for the 31-layer atlas renders)
# Runs Reader v2 (domenicor046/reader-v2, step 40000) through the same villa ink_9um inference entry point used
# elsewhere in this repo. 66-layer render, mesh at layer 33, 17-layer model window; both directions.
set -u
RENDER=$1; PREFIX=$2; shift 2
CKPT=D:/divesh-github/reader-v2-weights/reader-v2-step040000.pth
CENTER=${CENTER:-33}
for d in "$@"; do
  c=$((CENTER + d)); s=$((c - 8)); e=$((c + 9))
  out="${PREFIX}_d${d}.tif"
  if [ -f "$out" ] && [ -f "${PREFIX}_d${d}_reverse.tif" ]; then continue; fi
  D:/divesh-github/inkenv/Scripts/python.exe -m vesuvius.ink_detection.inference.infer "$RENDER" "$CKPT" "$out" \
     --overlap 0.5 --blend-mode hann --batch-size 4 --direction both --layer-start $s --layer-end $e \
     > "${PREFIX}_d${d}.log" 2>&1
  echo "done d=$d rc=$?"
done
echo RUN_DONE
