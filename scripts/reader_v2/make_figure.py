"""Side-by-side figure for one surface: raw CT (mesh layer), Reader v2 forward map, Reader v2 reverse map, and
the confident-text mask of reader-v2's text screen (pf>=180 & pr<=110, stroke-sized components) in each orientation.

Usage: make_figure.py <render.zarr> <prefix> <depth> <out.png> [title] [px_per_mm]
Run with D:/divesh-github/inkenv/Scripts/python.exe.
"""
import sys
import numpy as np, tifffile, zarr, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

render, prefix, d, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
title = sys.argv[5] if len(sys.argv) > 5 else prefix
um = 9.362
z = zarr.open(render, mode="r")
arr = z["0"] if "0" in z else z
ct = np.asarray(arr[arr.shape[0] // 2])
pf = tifffile.imread(f"{prefix}_d{d}.tif")
pr = tifffile.imread(f"{prefix}_d{d}_reverse.tif")
h, w = pf.shape
ct = ct[:h, :w]
valid = (ct > 0) & (pf > 0)
void = ct < (np.percentile(ct[valid], 15) if valid.any() else 0)


def confident(a, b, hi=180):
    m = valid & (a >= hi) & (b <= 110) & ~void
    lab, n = ndimage.label(m)
    if not n:
        return m
    s = ndimage.sum(m, lab, np.arange(1, n + 1))
    return np.isin(lab, np.flatnonzero((s >= 200) & (s <= 20000)) + 1)


mf, mr = confident(pf, pr), confident(pr, pf)
fig, ax = plt.subplots(4, 1, figsize=(14, 14 * 4 * h / w / 1.0 + 1.5))
ext = [0, w * um / 1000, h * um / 1000, 0]
ax[0].imshow(np.where(valid, ct, np.nan), cmap="gray", extent=ext); ax[0].set_title(f"{title}: raw CT at mesh layer (mm)")
for a, m, name in ((ax[1], pf, "forward"), (ax[2], pr, "reverse")):
    a.imshow(np.where(valid, m, np.nan), cmap="magma", vmin=0, vmax=255, extent=ext)
    a.set_title(f"Reader v2 {name} map")
ov = np.zeros((h, w, 3), np.float32)
ov[..., 0] = np.where(valid, ct, 0) / 255.0 * 0.6
ov[..., 1] = ov[..., 0]; ov[..., 2] = ov[..., 0]
ov[mf] = (1.0, 0.85, 0.0); ov[mr] = (0.0, 0.8, 1.0)
ax[3].imshow(ov, extent=ext)
ax[3].set_title(f"confident-text mask  yellow=fwd ({mf.sum() / valid.sum():.5f})  cyan=rev ({mr.sum() / valid.sum():.5f})")
for a in ax:
    a.set_xlabel("mm")
plt.tight_layout()
plt.savefig(out, dpi=90)
print(out, "valid mm^2:", round(valid.sum() * (um / 1000) ** 2, 1))
