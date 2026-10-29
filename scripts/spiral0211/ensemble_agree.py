"""Label-free ensemble signature per render/depth/direction: coherence + band std of the ensemble z-map, mean pairwise
correlation between the 4 models' z-maps (agreement), and p99 of the ensemble z. Works for 343 control and 0211.
Usage: ensemble_agree.py <ens_dir> <label> [depth list]   (run with the inkenv python)
"""
import glob, os, sys
import numpy as np, tifffile, cv2
from scipy.ndimage import gaussian_filter, binary_erosion

ens, label = sys.argv[1], sys.argv[2]
MODELS = ["seed42", "seed43", "dnative", "hecate"]

def load(m, d, rev):
    p = f"{ens}/{label}__{m}__d{d}" + ("_reverse" if rev else "")
    p += ("_native.tif" if m == "hecate" else ".tif")
    if not os.path.exists(p):
        return None
    return tifffile.imread(p)

depths = sorted({int(os.path.basename(p).split("__d")[1].split("_")[0].split(".")[0]) for p in glob.glob(f"{ens}/{label}__seed42__d*")})
print(f"{label}: d dir     n  coherence band_std  mean_pair_r  p99(z)  | per-model coherence")
for d in depths:
    for rev in (False, True):
        a = {m: load(m, d, rev) for m in MODELS}
        a = {m: v.astype(np.float32) for m, v in a.items() if v is not None}
        if len(a) < 2:
            continue
        valid = np.all([v > 0 for v in a.values()], axis=0)
        valid = binary_erosion(valid, iterations=30)
        if valid.sum() < 5000:
            continue
        z = {m: (v - v[valid].mean()) / (v[valid].std() + 1e-6) for m, v in a.items()}
        e = np.mean(list(z.values()), axis=0)
        def coh(x):
            xf = np.where(valid, x, 0)
            gy, gx = np.gradient(gaussian_filter(xf, 2.0))
            jxx, jyy, jxy = (gaussian_filter(q, 25) for q in (gx * gx, gy * gy, gx * gy))
            return float((np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy ** 2, 0)) / (jxx + jyy + 1e-6))[valid].mean())
        band = gaussian_filter(np.where(valid, e, 0), 1.5) - gaussian_filter(np.where(valid, e, 0), 12)
        ms = list(z)
        rs = [np.corrcoef(gaussian_filter(z[ms[i]], 4)[valid][::7], gaussian_filter(z[ms[j]], 4)[valid][::7])[0, 1]
              for i in range(len(ms)) for j in range(i + 1, len(ms))]
        print(f"{label}: {d:+3d} {'reverse' if rev else 'forward':8s} {len(a)}  {coh(e):.3f}     {band[valid].std():.3f}    {np.mean(rs):+.3f}      {np.percentile(e[valid], 99):5.2f} | "
              + " ".join(f"{m}:{coh(z[m]):.3f}" for m in ms), flush=True)
