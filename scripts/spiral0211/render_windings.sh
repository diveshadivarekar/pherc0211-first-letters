#!/usr/bin/env bash
# Usage: render_windings.sh <run_dir> <tag> <w1> <w2> ...   renders 66 layers per winding with VC3D (remote CT streaming into a shared cache)
set -u
RUN=$1; TAG=$2; shift 2
R=D:/divesh-github/virtual-unwrap/data/spiral
for w in "$@"; do
  n=$(printf "w%03d" $w)
  out=$R/render/${n}_${TAG}.zarr
  [ -d "$out/0/0" ] && { echo "skip $n"; continue; }
  "/c/Program Files/VC3D/bin/vc_render_tifxyz.exe" --volume $R/ct_cache/volume.zarr \
    --remote-url s3://vesuvius-challenge-open-data/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr \
    --group-idx 0 --scale 1.0 --segmentation "$RUN/meshes/fitted_$TAG/${n}_$TAG" --num-slices 66 --slice-step 1 \
    --cache-gb 8 --zarr-output "$out" --timeout 120 --log-path $R/render/${n}_${TAG}.log > /dev/null 2>&1
  echo "rendered $n"
done
echo RENDER_DONE
