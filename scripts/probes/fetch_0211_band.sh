#!/usr/bin/env bash
# Band-limited mirror of the PHerc 0211 inputs the spiral fit needs, z 7572-9328 (level-0 voxels).
set -u
PY=.venv/Scripts/python.exe
OUT=data/spiral/pherc0211_src
S=PHerc0211/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr
L=PHerc0211/representations/predictions/lasagna/20250821151803-lasagna-20260419180421
V=PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr
D="$PY -u scripts/download_zarr_band.py --out $OUT --strip PHerc0211/"
$D --src $S --level 1 --z0 3786 --z1 4664
$D --src $S --level 0 --z0 7572 --z1 9328
for a in nx ny grad_mag; do $D --src $L/PHerc0211_$a.ome.zarr --level 2 --z0 1893 --z1 2332; done
$D --src $V --level 4 --z0 473 --z1 583
echo ALL_DONE
