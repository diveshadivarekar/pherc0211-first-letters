#!/bin/bash
# Official Paris 4 3D ink model on the w01-w05 crops: infer (one at a time, GPU) then score. Run from the repo root.
set -e
PY=D:/divesh-github/inkenv/Scripts/python.exe
CK=D:/divesh-github/dino3d-weights/ckpt_78k_fullsup.pth
declare -A PFX=( [w01]=w01_20230702185753_r15 [w02]=w02_20231031143852 [w03]=w03_20231106155351 [w05]=w05_4424 )
for s in w01 w02 w03 w05; do
  d=data/real_scans/paris4/$s; o=data/processed/ink3d_official/${s}_prob.npy
  [ -f $o ] || $PY -u scripts/probes/ink3d_official.py infer $d/native_crop.h5 $CK $o > data/processed/ink3d_official/${s}_infer.log 2>&1
  $PY -u scripts/probes/ink3d_official.py score $o $d ${PFX[$s]} --tile-json $d/tile.json --out results/ink3d_official/$s.json > results/ink3d_official/${s}_score.log 2>&1
  echo done $s
done
