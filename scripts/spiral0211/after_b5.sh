#!/usr/bin/env bash
# After the b5 fit has released its memory: redo the invalid b3/b4 ensemble windings (renders hang when commit memory is exhausted by a running fit).
cd D:/divesh-github/virtual-unwrap
until grep -aq "POST_DONE b5\|b5 windings" data/processed/spiral_logs/post_b5.log 2>/dev/null; do sleep 20; done
RB3=D:/divesh-github/virtual-unwrap/$(ls -d data/spiral/fit_out/*_slice-9500-11000_*_b3k3)
RB4=D:/divesh-github/virtual-unwrap/$(ls -d data/spiral/fit_out/*_slice-6200-7700_*_b4k3)
bash scripts/spiral0211/render_ens_chain.sh "$RB3" b3k3 50 > data/processed/spiral_logs/chain_b3_fix.log 2>&1
bash scripts/spiral0211/render_ens_chain.sh "$RB4" b4k3 28 50 > data/processed/spiral_logs/chain_b4_fix.log 2>&1
echo FIX_DONE
