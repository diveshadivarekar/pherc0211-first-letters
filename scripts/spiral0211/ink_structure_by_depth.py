"""Band-pass texture statistics of ink maps across depth, forward vs reversed direction.

An ink signal should make ONE depth range stand out (high stroke-band energy, anisotropic, line-periodic)
relative to the other depths; flat profiles across depth mean the model just responds to papyrus texture.
Run with D:/divesh-github/inkenv/Scripts/python.exe (needs LZW TIFF support).
Usage: ink_structure_by_depth.py <prefix>   (reads <prefix>_d<D>.tif and <prefix>_d<D>_reverse.tif; d0 file is <prefix>_d0.tif)
"""
import glob, re, sys
import numpy as np, tifffile
from scipy.ndimage import gaussian_filter

prefix = sys.argv[1]
rows = []
for f in sorted(glob.glob(prefix + "_d*.tif")):
    m = re.search(r"_d(-?\d+)(_reverse)?\.tif$", f)
    if not m:
        continue
    a = tifffile.imread(f).astype(np.float32)
    a = a[100:-100, 100:-100]  # drop render edges
    lo = gaussian_filter(a, 12)
    band = gaussian_filter(a, 1.5) - lo           # stroke-scale band (~3-25 px = 30-250 um)
    coarse = a.std()
    gy, gx = np.gradient(gaussian_filter(a, 2.0))
    # structure tensor anisotropy (0 = isotropic blobs, 1 = parallel lines)
    jxx, jyy, jxy = (gaussian_filter(v, 25) for v in (gx * gx, gy * gy, gx * gy))
    tr = jxx + jyy
    det_term = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy ** 2, 0))
    coh = float(np.mean(det_term / (tr + 1e-6)))
    rows.append((int(m.group(1)), bool(m.group(2)), float(coarse), float(band.std()), coh))
print("depth direction  coarse_std  band_std  coherence")
for d, rev, cs, bs, coh in sorted(rows):
    print(f"{d:+4d}  {'reverse' if rev else 'forward':8s} {cs:8.2f} {bs:8.2f} {coh:8.3f}")
