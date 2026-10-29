#!/usr/bin/env bash
# Sequential, idempotent clean-up pass (runs only what is missing; one job at a time because concurrent jobs
# exhaust Windows commit memory and crash with ArrayMemoryError / DataLoader worker exits).
cd /d/divesh-github/virtual-unwrap
for m in z6720_w080 z7312_w080 z9120_w080; do
  zr=$(ls -d data/atlas0211/$m/surface-volumes/*.zarr | head -1)
  CENTER=15 bash scripts/reader_v2/run_reader_v2.sh "$zr" data/reader_v2/ink/atlas_$m 0 7 -7
done
for r in w040_b3k w055_b3k w070_b3k w028_b3k3; do
  bash scripts/reader_v2/run_reader_v2.sh data/spiral/render/$r.zarr data/reader_v2/ink/own_$r 0
done
echo CHAIN5_DONE
