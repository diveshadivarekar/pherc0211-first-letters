#!/usr/bin/env bash
# Hotspot follow-up chain (band z 7100-8600): for each cropped mesh name: L0 render (66 layers, verified 100%) -> snap ->
# 4-model ensemble at DEPTHS. One GPU job at a time. Usage: b6_chain.sh <crop_root> <name1> <name2> ...
set -u
CROPS=$1; shift
R=D:/divesh-github/virtual-unwrap/data/spiral
PY=D:/divesh-github/inkenv/Scripts/python.exe
DEPTHS=${DEPTHS:--4,-2,0,2,4}
render_ok() { tr '\r' '\n' < "$1" 2>/dev/null | grep -a "tile-row" | tail -n 1 | grep -aq "(100%)"; }
for n in "$@"; do
  out=$R/render/$n.zarr; log=$R/render/$n.log
  for try in 1 2 3; do
    render_ok "$log" && [ -d "$out/0/0" ] && break
    rm -rf "$out"
    "/c/Program Files/VC3D/bin/vc_render_tifxyz.exe" --volume $R/ct_cache/volume.zarr \
      --remote-url s3://vesuvius-challenge-open-data/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr \
      --group-idx 0 --scale 1.0 --segmentation "$CROPS/$n" --num-slices 66 --slice-step 1 \
      --cache-gb 8 --zarr-output "$out" --timeout 120 --log-path "$log" > /dev/null 2>&1
  done
  if ! render_ok "$log"; then echo "RENDER_FAILED $n"; continue; fi
  echo "[$(date +%H:%M)] rendered $n"
  [ -d $R/snap/$n.zarr ] || $PY -u D:/divesh-github/virtual-unwrap/scripts/spiral0211/snap_render.py $out $R/snap/$n.zarr > $R/snap/$n.log 2>&1
  echo "[$(date +%H:%M)] snapped $n: $(tail -n 1 $R/snap/$n.log)"
  $PY -u D:/divesh-github/virtual-unwrap/scripts/spiral0211/ensemble_run.py $R/snap/$n.zarr $R/snap_ens $n 9.362 $DEPTHS > $R/snap_ens/$n.log 2>&1
  echo "[$(date +%H:%M)] ensemble $n: $(ls $R/snap_ens | grep -c "^${n}__") files"
done
echo CHAIN_DONE
