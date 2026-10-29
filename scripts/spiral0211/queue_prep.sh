#!/usr/bin/env bash
# Usage: queue_prep.sh <ZB> <ZE> <TAG> <OUTER> <prev-prep-log-tag>: wait for previous prep, then run prep only.
set -u
ZB=$1; ZE=$2; TAG=$3; OUTER=$4; PREVTAG=$5
ROOT=D:/divesh-github/virtual-unwrap; cd $ROOT; LOG=$ROOT/data/processed/spiral_logs
until grep -aq "PREP_DONE $PREVTAG" $LOG/run_band_${PREVTAG}_prep.log 2>/dev/null; do sleep 20; done
PREP_ONLY=1 bash scripts/spiral0211/run_band.sh $ZB $ZE $TAG $OUTER > $LOG/run_band_${TAG}_prep.log 2>&1
