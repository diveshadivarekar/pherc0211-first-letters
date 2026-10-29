"""Find which fitted windings pass through given scroll points, and where (render pixels).

Usage: locate_xyz.py <mesh_dir> <tag> <x,y,z> [<x,y,z> ...]   prints, per point, the 3 nearest windings with distance (voxels)
and render pixel (row, col) = grid index / scale. Windings are <mesh_dir>/wNNN_<tag>/{x,y,z}.tif (tifxyz, -1 = hole).
"""
import glob, json, os, sys
import numpy as np, tifffile

mdir, tag = sys.argv[1], sys.argv[2]
pts = np.array([[float(v) for v in p.split(",")] for p in sys.argv[3:]])
res = {i: [] for i in range(len(pts))}
for d in sorted(glob.glob(f"{mdir}/w[0-9][0-9][0-9]_{tag}")):
    w = int(os.path.basename(d)[1:4])
    xyz = np.stack([tifffile.imread(f"{d}/{c}.tif") for c in "xyz"], -1).astype(np.float32)
    sc = json.load(open(f"{d}/meta.json"))["scale"][0]
    ok = xyz[..., 2] > 0
    for i, p in enumerate(pts):
        dist = np.where(ok, np.linalg.norm(xyz - p, axis=-1), np.inf)
        r, c = np.unravel_index(np.argmin(dist), dist.shape)
        res[i].append((dist[r, c], w, r / sc, c / sc))
for i, p in enumerate(pts):
    best = sorted(res[i])[:3]
    print(f"point {p.astype(int).tolist()}: " + "; ".join(f"w{w:03d} d={dd:.1f} px(row {r:.0f}, col {c:.0f})" for dd, w, r, c in best))
