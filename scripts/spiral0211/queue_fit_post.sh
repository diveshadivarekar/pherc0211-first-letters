#!/usr/bin/env bash
# Usage: queue_fit_post.sh <ZB> <ZE> <TAG> <OUTER> <wait-for-log-pattern-file>  : wait for prep + previous GPU job, then fit and post_fit.
set -u
ZB=$1; ZE=$2; TAG=$3; OUTER=$4; PREV=$5
ROOT=D:/divesh-github/virtual-unwrap; cd $ROOT; LOG=$ROOT/data/processed/spiral_logs
until grep -aq "PREP_DONE $TAG" $LOG/run_band_${TAG}_prep.log 2>/dev/null; do sleep 20; done
until grep -aq "POST_DONE" $PREV 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M)] fit $TAG"
ZB=$ZB ZE=$ZE OUTER=$OUTER DATASET=pherc0211_$TAG bash scripts/spiral0211/run_fit.sh $ROOT/data/spiral/pherc0211_$TAG/spiral-scroll.json ${STEPS:-1000} ${TAG}k3 > $LOG/fit_${TAG}.log 2>&1
bash scripts/spiral0211/post_fit.sh $ZB $ZE $TAG > $LOG/post_${TAG}.log 2>&1
echo "[$(date +%H:%M)] QUEUE_DONE $TAG"
