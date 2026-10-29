"""Control vs 0211: per render, ensemble z-map (mean of 4 per-model z) at d=0: forward vs reverse coherence, p99, mean pairwise
model correlation, and corr(forward, reverse) of the ensemble. Run with the inkenv python."""
import numpy as np, tifffile
from scipy.ndimage import gaussian_filter, binary_erosion
MODELS = ("seed42", "seed43", "dnative", "hecate")
def maps(prefix, d, rev):
    out = []
    for m in MODELS:
        p = f"{prefix}__{m}__d{d}" + ("_reverse" if rev else "") + ("_native.tif" if m == "hecate" else ".tif")
        out.append(tifffile.imread(p).astype(np.float32))
    return out
def coh(x, v):
    xf = np.where(v, x, 0); gy, gx = np.gradient(gaussian_filter(xf, 2.0))
    jxx, jyy, jxy = (gaussian_filter(q, 25) for q in (gx * gx, gy * gy, gx * gy))
    return float((np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy ** 2, 0)) / (jxx + jyy + 1e-6))[v].mean())
def stats(prefix, d):
    res = {}
    A = {r: maps(prefix, d, r) for r in (False, True)}
    v = binary_erosion(np.all([x > 0 for r in A for x in A[r]], 0), iterations=30)
    for r in (False, True):
        z = [(x - x[v].mean()) / x[v].std() for x in A[r]]
        e = np.mean(z, 0)
        rs = [np.corrcoef(gaussian_filter(z[i], 4)[v][::7], gaussian_filter(z[j], 4)[v][::7])[0, 1] for i in range(4) for j in range(i + 1, 4)]
        res[r] = dict(e=e, coh=coh(e, v), p99=float(np.percentile(e[v], 99)), pair=float(np.mean(rs)))
    c = np.corrcoef(gaussian_filter(res[False]["e"], 4)[v][::7], gaussian_filter(res[True]["e"], 4)[v][::7])[0, 1]
    return res, float(c)
import sys
rows = [("343 control d=0", "D:/divesh-github/virtual-unwrap/data/control_343/ens/concat", 0)]
rows += [(f"0211 {w} d=0", f"D:/divesh-github/virtual-unwrap/data/spiral/ens/{w}", 0) for w in ("w020", "w035", "w045", "w055", "w065")]
if len(sys.argv) > 1:  # name=prefix pairs
    rows = [(a.split("=")[0], a.split("=", 1)[1], 0) for a in sys.argv[1:]]
print(f"{'render':16s} coh fw/rv      p99 fw/rv    model-agree fw/rv   corr(fw,rv)")
for name, pre, d in rows:
    try:
        r, c = stats(pre, d)
    except Exception as e:
        print(f"{name:16s} (missing: {type(e).__name__})"); continue
    print(f"{name:16s} {r[False]['coh']:.3f}/{r[True]['coh']:.3f}   {r[False]['p99']:.2f}/{r[True]['p99']:.2f}   {r[False]['pair']:+.2f}/{r[True]['pair']:+.2f}      {c:+.2f}", flush=True)
