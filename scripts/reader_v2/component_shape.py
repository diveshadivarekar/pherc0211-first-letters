"""Shape statistics of the confident-text components (reader-v2 text-screen definition) for one map pair.

Text in a rolled scroll runs along the winding direction = image x axis of a tifxyz render, so ink strokes form
horizontal rows of blobs a few mm tall. Components that are tall and narrow (vertical stripes spanning most of the
page height) are what crossings / sheet edges / reconstruction streaks produce instead.

Per orientation (forward map as ink vs reverse as ink, and vice versa) prints: n components, area share in
components whose major axis is within 30 deg of horizontal / vertical, median and max height (mm), and the
fraction of components taller than 3 mm.

Usage: component_shape.py <render.zarr> <prefix> <depth> [um=9.362]   (inkenv python)
"""
import sys
import numpy as np, tifffile, zarr
from scipy import ndimage

render, prefix, d = sys.argv[1], sys.argv[2], int(sys.argv[3])
um = float(sys.argv[4]) if len(sys.argv) > 4 else 9.362
z = zarr.open(render, mode="r")
arr = z["0"] if "0" in z else z
ct = np.asarray(arr[arr.shape[0] // 2])
pf = tifffile.imread(f"{prefix}_d{d}.tif"); pr = tifffile.imread(f"{prefix}_d{d}_reverse.tif")
h, w = pf.shape
ct = ct[:h, :w]
valid = (ct > 0) & (pf > 0)
void = ct < np.percentile(ct[valid], 15)


def shape_stats(a, b, hi=180):
    m = valid & (a >= hi) & (b <= 110) & ~void
    lab, n = ndimage.label(m)
    out = dict(n=0, area_h=0.0, area_v=0.0, med_h_mm=0.0, max_h_mm=0.0, tall_frac=0.0, area_frac=0.0)
    if not n:
        return out
    idx = np.arange(1, n + 1)
    area = ndimage.sum(m, lab, idx)
    keep = idx[(area >= 200) & (area <= 20000)]
    if not len(keep):
        return out
    ys, xs = np.indices(m.shape)
    tot = hor = ver = 0.0
    heights = []
    for k in keep:
        sl = ndimage.find_objects((lab == k).astype(np.uint8))[0]
        sub = lab[sl] == k
        yy, xx = np.nonzero(sub)
        a_ = len(yy)
        cov = np.cov(np.vstack([xx, yy]))
        evals, evecs = np.linalg.eigh(cov)
        vx, vy = evecs[:, 1]
        ang = abs(np.degrees(np.arctan2(vy, vx)))
        ang = min(ang, 180 - ang)
        elong = np.sqrt(evals[1] / max(evals[0], 1e-6))
        tot += a_
        if elong > 1.5:
            if ang < 30: hor += a_
            elif ang > 60: ver += a_
        heights.append((yy.max() - yy.min() + 1) * um / 1000)
    heights = np.array(heights)
    out.update(n=len(keep), area_h=hor / tot, area_v=ver / tot, med_h_mm=float(np.median(heights)),
               max_h_mm=float(heights.max()), tall_frac=float((heights > 3).mean()), area_frac=tot / valid.sum())
    return out


for name, a, b in (("forward-as-ink", pf, pr), ("reverse-as-ink", pr, pf)):
    s = shape_stats(a, b)
    print(f"{prefix.split('/')[-1]} d={d:+d} {name:15s} n={s['n']:3d} area_frac={s['area_frac']:.5f} "
          f"horiz_area={s['area_h']:.2f} vert_area={s['area_v']:.2f} median_h={s['med_h_mm']:.2f}mm "
          f"max_h={s['max_h_mm']:.2f}mm tall(>3mm)={s['tall_frac']:.2f}")
