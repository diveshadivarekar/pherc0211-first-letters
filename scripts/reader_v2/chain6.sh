#!/usr/bin/env bash
# Deep depth sweep on the 66-layer re-render of atlas mesh PHerc0211 z7920_w040 (mesh at layer 33, window 17 layers).
cd /d/divesh-github/virtual-unwrap
CENTER=33 bash scripts/reader_v2/run_reader_v2.sh data/atlas0211/z7920_w040/render66.zarr data/reader_v2/ink/deep_z7920_w040 "$@"
echo CHAIN6_DONE
