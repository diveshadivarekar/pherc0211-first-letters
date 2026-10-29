"""Run the winner-commit (0fb38c45) extract_surface_tracks.py on the local 0211 level-0 band.

The upstream script is configured through module globals; this wrapper sets them and calls main().
Run with the spiral-fitting venv: D:/divesh-github/villa/spiral-fitting/.venv/Scripts/python.exe
Globals are set at import time so spawned workers (Windows) see them; main() runs only in __main__.
"""
import os
import sys

UP = "D:/divesh-github/villa_0fb38c45/volume-cartographer/scripts/spiral"
sys.path.insert(0, UP)
import extract_surface_tracks as e  # noqa: E402

ROOT = "D:/divesh-github/virtual-unwrap/data/spiral"
Z0, Z1 = int(os.environ.get("TRK_Z0", 7636)), int(os.environ.get("TRK_Z1", 9264))  # env is inherited by spawned workers
e.predictions_zarr_path = (f"{ROOT}/pherc0211_src/representations/predictions/surfaces/"
                           "20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr/0")
e.tracks_dbm_path = f"{ROOT}/{os.environ.get('TRK_DATASET', 'pherc0211')}/tracks/pherc0211_z{Z0}-{Z1}.dbm"
e.z_min, e.z_max = Z0, Z1
if os.environ.get("TRK_FAST"):  # half-density ribbons/slabs: ~2x faster, thinner track coverage (screening quality)
    e.z_chunk_stride_h, e.yx_stride_v = 32, 128

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    os.makedirs(os.path.dirname(e.tracks_dbm_path), exist_ok=True)
    print({k: getattr(e, k) for k in ("downsample_factor", "z_chunk_depth_h", "z_chunk_stride_h",
                                      "yx_chunk_thickness_v", "yx_stride_v", "dust_threshold",
                                      "max_area_threshold", "path_mode", "write_native_packed_store")}, flush=True)
    import numpy as np
    np.random.seed(0)
    e.main()
