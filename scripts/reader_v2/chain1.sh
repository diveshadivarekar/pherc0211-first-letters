#!/usr/bin/env bash
# First chain: 343 control + one own-mesh winding + five atlas surfaces, d=0 (d=+1 for the control).
cd /d/divesh-github/virtual-unwrap
bash scripts/reader_v2/run_reader_v2.sh data/control_343/render/concat_crop.zarr data/reader_v2/ink/ctrl343 1
bash scripts/reader_v2/run_reader_v2.sh data/spiral/render/w045_cw600.zarr data/reader_v2/ink/w045 0
for m in z7920_w040 z6720_w080 z9120_w080 z7312_w080 z9712_w080; do
  zr=$(ls -d data/atlas0211/$m/surface-volumes/*.zarr | head -1)
  CENTER=15 bash scripts/reader_v2/run_reader_v2.sh "$zr" data/reader_v2/ink/atlas_$m 0
done
echo CHAIN1_DONE
