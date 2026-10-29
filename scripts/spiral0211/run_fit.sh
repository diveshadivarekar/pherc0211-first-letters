#!/usr/bin/env bash
# Usage: [ZB=7700 ZE=9200 OUTER=72 DATASET=pherc0211] run_fit.sh <scroll-spec json> <steps> <tag>
# Winner's overrides (PHerc 343 README): disable patches, shell weights 0, phase spacing, 30k steps.
set -eu
SPEC=$1; STEPS=$2; TAG=$3
ZB=${ZB:-7700}; ZE=${ZE:-9200}; OUTER=${OUTER:-72}; DATASET=${DATASET:-pherc0211}
export PYTHONPATH="D:/divesh-github/virtual-unwrap/scripts/spiral0211/shim;D:/divesh-github/villa_0fb38c45/volume-cartographer/scripts/spiral"
export FIT_SPIRAL_OUT_DIR=D:/divesh-github/virtual-unwrap/data/spiral/fit_out
export FIT_SPIRAL_CACHE_DIR=D:/divesh-github/virtual-unwrap/data/spiral/fit_cache
export FIT_SPIRAL_RUN_TAG=$TAG
export WANDB_MODE=disabled
export FIT_SPIRAL_CONFIG_OVERRIDES="{\"z_begin\":$ZB,\"z_end\":$ZE,\"input_disable_patches\":true,\"loss_weight_shell_outer\":0,\"loss_weight_shell_patch_radius\":0,\"dense_spacing_mode\":\"phase\",\"shell_outer_winding_idx\":$OUTER,\"track_crossing_mode\":\"count\",\"optimizer_num_training_steps\":$STEPS}"
cd /d/divesh-github/villa_0fb38c45/volume-cartographer/scripts/spiral
exec /d/divesh-github/villa/spiral-fitting/.venv/Scripts/python.exe -u fit_spiral.py \
  --dataset D:/divesh-github/virtual-unwrap/data/spiral/$DATASET \
  --scroll-spec "$SPEC" --cache "$FIT_SPIRAL_CACHE_DIR"
