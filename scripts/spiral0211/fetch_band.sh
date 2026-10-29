#!/usr/bin/env bash
# Band-limited mirror of the 0211 inputs for the spiral fit. Usage: fetch_band.sh <z0> <z1>   (level-0 voxels, include ~128 margin)
set -eu
Z0=$1; Z1=$2
cd D:/divesh-github/virtual-unwrap
PY=.venv/Scripts/python.exe
OUT=data/spiral/pherc0211_src
S=PHerc0211/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr
L=PHerc0211/representations/predictions/lasagna/20250821151803-lasagna-20260419180421
V=PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr
D="$PY -u scripts/download_zarr_band.py --out $OUT --strip PHerc0211/"
$D --src $S --level 1 --z0 $((Z0/2)) --z1 $((Z1/2 + 1))
$D --src $S --level 0 --z0 $Z0 --z1 $Z1
for a in nx ny; do $D --src $L/PHerc0211_$a.ome.zarr --level 2 --z0 $((Z0/4)) --z1 $((Z1/4 + 1)); done
$D --src $V --level 4 --z0 $((Z0/16)) --z1 $((Z1/16 + 1))
echo ALL_DONE
