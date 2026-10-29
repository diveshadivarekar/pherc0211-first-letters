"""Fraction of fitted-mesh nodes that lie on (within r voxels of) the m7 surface prediction.

Uses the level-2 prediction (4x down, whole scroll mirrored locally) restricted to the fit z-band.
Usage: mesh_on_surface.py <run_dir> <comma-separated winding numbers>
"""
import glob, sys
import numpy as np, tifffile, zarr
from scipy import ndimage as ndi

run, windings = sys.argv[1], [int(w) for w in sys.argv[2].split(",")]
S = 4
pred = zarr.open("D:/divesh-github/virtual-unwrap/data/real_scans/pherc0211/representations/predictions/surfaces/"
                 "20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr/2", mode="r")
import os
ZB, ZE = int(os.environ.get("ZB", 7700)), int(os.environ.get("ZE", 9200))
z0, z1 = ZB // S, ZE // S
vol = np.asarray(pred[z0:z1]) > 127
vols = {r: (ndi.binary_dilation(vol, iterations=r) if r else vol) for r in (0, 1, 2)}
for w in windings:
    d = [p for p in glob.glob(f"{run}/meshes/*/w{w:03d}_*") if "spliced" not in p][0]
    x, y, z = (tifffile.imread(f"{d}/{c}.tif").astype(np.float64) for c in "xyz")
    ok = (x > 0) & (z >= ZB) & (z < ZE)
    pts = np.stack([z[ok], y[ok], x[ok]], 1) / S
    idx = np.round(pts).astype(int); idx[:, 0] -= z0
    inb = (idx[:, 0] >= 0) & (idx[:, 0] < vol.shape[0]) & (idx[:, 1] >= 0) & (idx[:, 1] < vol.shape[1]) & (idx[:, 2] >= 0) & (idx[:, 2] < vol.shape[2])
    idx = idx[inb]
    h = {r: vols[r][idx[:, 0], idx[:, 1], idx[:, 2]].mean() * 100 for r in vols}
    # grid step 20 voxels, 9.362 um voxels
    area_cm2 = ok.sum() * (20 * 0.009362) ** 2 / 100
    print(f"w{w:03d}: {ok.sum()} nodes, {area_cm2:.1f} cm2; on predicted surface within 0/4/8 vox: {h[0]:.0f}/{h[1]:.0f}/{h[2]:.0f}%", flush=True)
