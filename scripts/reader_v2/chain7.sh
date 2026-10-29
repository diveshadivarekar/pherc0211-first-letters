#!/usr/bin/env bash
# Sheet-tracked stacks of mesh z7920_w040 (33 layers, tracked sheet at layer 16), one job at a time.
cd /d/divesh-github/virtual-unwrap
Z=data/atlas0211/z7920_w040
CENTER=16 bash scripts/reader_v2/run_reader_v2.sh $Z/trackB_center.zarr data/reader_v2/ink/trackB_center 0 -4 -8
CENTER=16 bash scripts/reader_v2/run_reader_v2.sh $Z/trackB_edge.zarr   data/reader_v2/ink/trackB_edge 0 4
CENTER=16 bash scripts/reader_v2/run_reader_v2.sh $Z/trackA_center.zarr data/reader_v2/ink/trackA_center 0
echo CHAIN7_DONE
