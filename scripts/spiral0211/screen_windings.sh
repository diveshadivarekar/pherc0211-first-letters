#!/usr/bin/env bash
# Wait for renders, then ink_9um depth screen (-8,0,+8) on each rendered winding. Usage: screen_windings.sh <tag> <w1> <w2> ...
set -u
TAG=$1; shift
R=D:/divesh-github/virtual-unwrap/data/spiral
CK=D:/divesh-github/ink9um/hybrid_3d2d-seed42/step-075000.pth
for w in "$@"; do
  n=$(printf "w%03d" $w)
  until [ -d "$R/render/${n}_${TAG}.zarr/0/0" ] && grep -q "rendered $n" /d/divesh-github/virtual-unwrap/data/processed/spiral_logs/render_more.log; do sleep 20; done
  bash /d/divesh-github/virtual-unwrap/scripts/spiral0211/ink_depth_sweep.sh "$R/render/${n}_${TAG}.zarr" "$R/ink/${n}_${TAG}" $CK -8 0 8 > /dev/null
  echo "screened $n"
done
echo SCREEN_DONE
