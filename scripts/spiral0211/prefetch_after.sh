#!/usr/bin/env bash
# Wait for the band-3 fetch to finish, then pre-fetch next bands while CPU/GPU work on band 3.
cd D:/divesh-github/virtual-unwrap
until grep -aq ALL_DONE data/processed/spiral_logs/fetch_b3.log 2>/dev/null; do sleep 20; done
bash scripts/spiral0211/fetch_band.sh 6072 7828 > data/processed/spiral_logs/fetch_b4.log 2>&1
bash scripts/spiral0211/fetch_band.sh 10872 12628 > data/processed/spiral_logs/fetch_b5.log 2>&1
echo PREFETCH_DONE
