#!/usr/bin/env bash
# Score every finished Reader v2 map pair, then write the merged markdown table.
cd /d/divesh-github/virtual-unwrap
PY=D:/divesh-github/inkenv/Scripts/python.exe
S=scripts/reader_v2/score_reader_v2.py
O=data/reader_v2
rm -f $O/score_all.csv
# positive control
$PY $S data/control_343/render/concat_crop.zarr $O/score_all.csv $O/ink/ctrl343 > /dev/null
# own meshes
$PY $S data/spiral/render/w045_cw600.zarr $O/score_all.csv $O/ink/w045 > /dev/null
for r in w020_senseCW w030_senseCW w035_senseCW w040_senseCW w050_senseCW w055_senseCW w065_senseCW w025_b3k w040_b3k w055_b3k w070_b3k w028_b3k3; do
  ls $O/ink/own_${r}_d*_reverse.tif > /dev/null 2>&1 || continue
  $PY $S data/spiral/render/$r.zarr $O/score_all.csv $O/ink/own_$r > /dev/null
done
# atlas surfaces
for m in z7920_w040 z6720_w080 z9120_w080 z7312_w080; do
  zr=$(ls -d data/atlas0211/$m/surface-volumes/*.zarr | head -1)
  $PY $S "$zr" $O/score_all.csv $O/ink/atlas_$m > /dev/null
done
$PY scripts/reader_v2/summarize.py $O/score_table.md $O/score_all.csv
