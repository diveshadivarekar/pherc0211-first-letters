#!/usr/bin/env bash
# Second chain: Reader v2 at d=0 on the other already-rendered own-mesh windings (66-layer, CENTER=33).
cd /d/divesh-github/virtual-unwrap
for r in w020_senseCW w030_senseCW w035_senseCW w040_senseCW w050_senseCW w055_senseCW w065_senseCW \
         w025_b3k w040_b3k w055_b3k w070_b3k w028_b3k3; do
  bash scripts/reader_v2/run_reader_v2.sh data/spiral/render/$r.zarr data/reader_v2/ink/own_$r 0
done
echo CHAIN2_DONE
