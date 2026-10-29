"""Crop a tifxyz mesh to a column window around the grid point nearest a scroll xyz (keeps all rows).

Usage: crop_tifxyz.py <mesh_dir> <out_dir> <x,y,z> <half_width_px>
half_width_px is in render pixels (grid cols = px * scale). Prints the crop's column range, the anchor's render pixel
in the cropped render (row, col) and its distance from the point. Holes (-1) are kept as they are.
"""
import json, os, sys
import numpy as np, tifffile

src, out, p, hw = sys.argv[1], sys.argv[2], np.array([float(v) for v in sys.argv[3].split(",")]), float(sys.argv[4])
meta = json.load(open(f"{src}/meta.json"))
sc = meta["scale"][0]
xyz = np.stack([tifffile.imread(f"{src}/{c}.tif") for c in "xyz"], -1)
ok = xyz[..., 2] > 0
dist = np.where(ok, np.linalg.norm(xyz - p, axis=-1), np.inf)
r, c = np.unravel_index(np.argmin(dist), dist.shape)
h = int(round(hw * sc))
c0, c1 = max(0, c - h), min(xyz.shape[1], c + h + 1)
os.makedirs(out, exist_ok=True)
for i, n in enumerate("xyz"):
    tifffile.imwrite(f"{out}/{n}.tif", np.ascontiguousarray(xyz[:, c0:c1, i]))
sub = xyz[:, c0:c1][ok[:, c0:c1]]
meta["bbox"] = [sub.min(0).tolist(), sub.max(0).tolist()]
meta["uuid"] = os.path.basename(out.rstrip("/\\"))
meta["area_cm2"] = meta.get("area_cm2", 0) * (c1 - c0) / xyz.shape[1]
meta["area_vx2"] = meta.get("area_vx2", 0) * (c1 - c0) / xyz.shape[1]
json.dump(meta, open(f"{out}/meta.json", "w"), indent=4)
print(f"cols {c0}-{c1} of {xyz.shape[1]}; anchor render px (row {r / sc:.0f}, col {(c - c0) / sc:.0f}); dist {dist[r, c]:.1f} vox")
