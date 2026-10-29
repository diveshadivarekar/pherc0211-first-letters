#!/usr/bin/env bash
# 4-model ensemble at d=0 on level-0 renders of crude CW-600 meshes (66 layers, mesh at layer 33).
set -u
R=D:/divesh-github/virtual-unwrap/data/spiral
for z in w020_senseCW w035_senseCW w045_cw600 w055_senseCW w065_senseCW; do
  lab=${z%%_*}
  D:/divesh-github/inkenv/Scripts/python.exe -u D:/divesh-github/virtual-unwrap/scripts/spiral0211/ensemble_run.py $R/render/$z.zarr $R/ens $lab 9.362 0 > /dev/null 2>&1
  echo "done $lab"
done
echo ENS0211_DONE
