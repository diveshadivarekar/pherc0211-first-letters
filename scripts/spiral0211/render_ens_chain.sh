#!/usr/bin/env bash
# Usage: render_ens_chain.sh <run_dir> <tag> <w1> <w2> ... : L0 render (66 layers) -> verify render complete -> 4-model ensemble d=0.
# A render counts as complete only if its log ends with the final tile-row at 100%; incomplete renders are redone (max 3 tries).
set -u
RUN=$1; TAG=$2; shift 2
R=D:/divesh-github/virtual-unwrap/data/spiral
render_ok() { tr '\r' '\n' < "$1" 2>/dev/null | grep -a "tile-row" | tail -n 1 | grep -aq "(100%)"; }
for w in "$@"; do
  n=$(printf "w%03d" $((10#$w)))
  out=$R/render/${n}_${TAG}.zarr; log=$R/render/${n}_${TAG}.log
  for try in 1 2 3; do
    render_ok "$log" && [ -d "$out/0/0" ] && break
    rm -rf "$out"
    "/c/Program Files/VC3D/bin/vc_render_tifxyz.exe" --volume $R/ct_cache/volume.zarr \
      --remote-url s3://vesuvius-challenge-open-data/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr \
      --group-idx 0 --scale 1.0 --segmentation "$RUN/meshes/fitted_$TAG/${n}_$TAG" --num-slices 66 --slice-step 1 \
      --cache-gb 8 --zarr-output "$out" --timeout 120 --log-path "$log" > /dev/null 2>&1
  done
  if render_ok "$log"; then
    D:/divesh-github/inkenv/Scripts/python.exe -u D:/divesh-github/virtual-unwrap/scripts/spiral0211/ensemble_run.py $out $R/ens ${n}_${TAG} 9.362 0 > /dev/null 2>&1
    echo "done $n"
  else
    echo "RENDER_FAILED $n"
  fi
done
echo CHAIN_DONE
