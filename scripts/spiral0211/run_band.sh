#!/usr/bin/env bash
# End-to-end band run: download -> tracks -> normals copy -> SDT -> 3000-step fit -> L1 single-model screen + L0 4-model ensemble.
# Usage: run_band.sh <ZB> <ZE> <tag> <outer_idx>      e.g. run_band.sh 9500 11000 b3 74
set -u
ZB=$1; ZE=$2; TAG=$3; OUTER=$4
STEPS=${STEPS:-1000}; export TRK_FAST=${TRK_FAST:-1}; STRIDE=${STRIDE:-6}
ROOT=D:/divesh-github/virtual-unwrap
cd $ROOT
DS=pherc0211_$TAG
LOG=$ROOT/data/processed/spiral_logs
Z0=$((ZB-128)); Z1=$((ZE+128)); T0=$((ZB-64)); T1=$((ZE+64))
PYV=/d/divesh-github/villa/spiral-fitting/.venv/Scripts/python.exe
if [ -z "${SKIP_PREP:-}" ]; then
echo "[$(date +%H:%M)] fetch $Z0-$Z1"
bash scripts/spiral0211/fetch_band.sh $Z0 $Z1 > $LOG/fetch_$TAG.log 2>&1
mkdir -p data/spiral/$DS/lasagna_inputs data/spiral/$DS/tracks
cp data/spiral/pherc0211/umbilicus.json data/spiral/$DS/
python - <<PY
import json
d=json.load(open("data/spiral/pherc0211/spiral-scroll.json")); d["paths"]={"tracks_dbm":"tracks/pherc0211_z$T0-$T1.dbm"}
json.dump(d,open("data/spiral/$DS/spiral-scroll.json","w"),indent=2)
PY
echo "[$(date +%H:%M)] tracks $T0-$T1"
TRK_Z0=$T0 TRK_Z1=$T1 TRK_DATASET=$DS $PYV -u scripts/spiral0211/run_tracks.py > $LOG/tracks_$TAG.log 2>&1 &
TPID=$!
L=data/spiral/pherc0211_src/representations/predictions/lasagna/20250821151803-lasagna-20260419180421
cp -r $L/PHerc0211_nx.ome.zarr data/spiral/$DS/lasagna_inputs/las_008_nx.ome.zarr
cp -r $L/PHerc0211_ny.ome.zarr data/spiral/$DS/lasagna_inputs/las_008_ny.ome.zarr
SRC=$ROOT/data/spiral/pherc0211_src
echo "[$(date +%H:%M)] sdt"
( cd /d/divesh-github/villa_0fb38c45/volume-cartographer/scripts/spiral && PYTHONPATH=. $PYV -u make_surf_sdt.py \
  --surf $SRC/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr --group 1 --threshold 128 \
  --out $ROOT/data/spiral/$DS/lasagna_inputs/las_008_surf_sdt.ome.zarr --working-voxel-um 9.362 --z-range $T0:$T1 \
  --ct $SRC/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr --ct-group 4 --ct-dilate 1 --no-ct-zero --workers 3 ) > $LOG/sdt_$TAG.log 2>&1
wait $TPID
fi
[ -n "${PREP_ONLY:-}" ] && { echo "[$(date +%H:%M)] PREP_DONE $TAG"; exit 0; }
echo "[$(date +%H:%M)] fit"
ZB=$ZB ZE=$ZE OUTER=$OUTER DATASET=$DS bash scripts/spiral0211/run_fit.sh $ROOT/data/spiral/$DS/spiral-scroll.json $STEPS ${TAG}k3 > $LOG/fit_${TAG}.log 2>&1
RUN=$ROOT/$(ls -d data/spiral/fit_out/*_slice-${ZB}-${ZE}_*_${TAG}k3 | head -1)
MESHDIR=$RUN/meshes/fitted_${TAG}k3
WMIN=$(ls $MESHDIR | grep -v spliced | sed -E 's/^w([0-9]+)_.*/\1/' | sort -n | head -1)
WMAX=$(ls $MESHDIR | grep -v spliced | sed -E 's/^w([0-9]+)_.*/\1/' | sort -n | tail -1)
echo "[$(date +%H:%M)] windings $WMIN-$WMAX in $RUN"
ORDER=$(python -c "
a=list(range(int('$WMIN')+4,int('$WMAX')+1,int('$STRIDE')))
print(','.join(map(str,a)))")
echo "[$(date +%H:%M)] screen + ensemble"
LOCAL_VOL=$ROOT/data/real_scans/pherc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr \
  D:/divesh-github/inkenv/Scripts/python.exe -u scripts/spiral0211/screen_L1.py "$RUN" ${TAG}k3 "$ORDER" 0 0 > $LOG/screen_L1_$TAG.log 2>&1 &
W1=$((WMIN+20)); W2=$((WMIN+42))
bash scripts/spiral0211/render_ens_chain.sh "$RUN" ${TAG}k3 $W1 $W2 > $LOG/chain_$TAG.log 2>&1
wait
echo "[$(date +%H:%M)] BAND_DONE $TAG"
