#!/usr/bin/env bash
# Depth extremes (|d|=7, the render's limit) on the other atlas surfaces; run two instances in parallel with different surface lists.
cd /d/divesh-github/virtual-unwrap
for m in "$@"; do
  zr=$(ls -d data/atlas0211/$m/surface-volumes/*.zarr | head -1)
  CENTER=15 bash scripts/reader_v2/run_reader_v2.sh "$zr" data/reader_v2/ink/atlas_$m 7 -7
done
echo CHAIN4_DONE
