#!/usr/bin/env bash
# Depth sweep (spatial-depth control) on the strongest atlas candidate; 31-layer render, mesh at layer 15, so |d| <= 7.
cd /d/divesh-github/virtual-unwrap
zr=$(ls -d data/atlas0211/z7920_w040/surface-volumes/*.zarr | head -1)
CENTER=15 bash scripts/reader_v2/run_reader_v2.sh "$zr" data/reader_v2/ink/atlas_z7920_w040 -7 -4 4 7
echo CHAIN3_DONE
