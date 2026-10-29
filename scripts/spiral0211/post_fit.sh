#!/usr/bin/env bash
# After a band's fit: L1 single-model screen (every STRIDE-th winding) + L0 4-model ensemble on two windings.
# Usage: post_fit.sh <ZB> <ZE> <TAG>      (waits for the fit's meshes to be written)
set -u
ZB=$1; ZE=$2; TAG=$3; STRIDE=${STRIDE:-6}
ROOT=D:/divesh-github/virtual-unwrap; cd $ROOT; LOG=$ROOT/data/processed/spiral_logs
until ls -d data/spiral/fit_out/*_slice-${ZB}-${ZE}_*_${TAG}k3/meshes/fitted_${TAG}k3/w*_${TAG}k3 > /dev/null 2>&1 && tr '\r' '\n' < $LOG/fit_${TAG}.log | grep -aq "Writing final mesh windings.*100.0%"; do sleep 15; done
sleep 5
RUN=$ROOT/$(ls -d data/spiral/fit_out/*_slice-${ZB}-${ZE}_*_${TAG}k3 | head -1)
MESHDIR=$RUN/meshes/fitted_${TAG}k3
WMIN=$(ls $MESHDIR | grep -v spliced | sed -E 's/^w([0-9]+)_.*/\1/' | sort -n | head -1)
WMAX=$(ls $MESHDIR | grep -v spliced | sed -E 's/^w([0-9]+)_.*/\1/' | sort -n | tail -1)
ORDER=$(python -c "print(','.join(map(str,range(int('$WMIN')+4,int('$WMAX')+1,int('$STRIDE')))))")
echo "[$(date +%H:%M)] $TAG windings $WMIN-$WMAX; screen order $ORDER"
LOCAL_VOL=$ROOT/data/real_scans/pherc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr \
  D:/divesh-github/inkenv/Scripts/python.exe -u scripts/spiral0211/screen_L1.py "$RUN" ${TAG}k3 "$ORDER" 0 0 > $LOG/screen_L1_$TAG.log 2>&1 &
bash scripts/spiral0211/render_ens_chain.sh "$RUN" ${TAG}k3 $((WMIN+20)) $((WMIN+42)) > $LOG/chain_$TAG.log 2>&1
wait
echo "[$(date +%H:%M)] POST_DONE $TAG"
