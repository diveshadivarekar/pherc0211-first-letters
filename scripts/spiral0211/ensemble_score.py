"""Score the 4-model ensemble on the PHerc 343 control (winner's letter boxes) per depth and direction.

Per model z-score over valid pixels (a>0), ensemble = mean of z (winner's rule). Reports AUC letters-vs-rest for each
model and the ensemble, shifted-box null AUCs, and band-pass/coherence signature of the ensemble z-map.
Run with D:/divesh-github/inkenv/Scripts/python.exe.  Usage: ensemble_score.py <ens_dir> <label> [SC]
"""
import glob, json, os, sys
import numpy as np, tifffile, cv2
from scipy.ndimage import gaussian_filter

ens, label = sys.argv[1], sys.argv[2]
SC = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
X0, Y0 = 1500 / SC, 600 / SC
boxes = json.load(open("D:/divesh-github/virtual-unwrap/data/control_343/winner_files/letter_boxes.json"))["letters"]
MODELS = ["seed42", "seed43", "dnative", "hecate"]

def load(model, d, rev):
    suf = "_reverse" if rev else ""
    p = f"{ens}/{label}__{model}__d{d}{suf}"
    if model == "hecate":
        p += "_native.tif" if os.path.exists(p + "_native.tif") else ".png"
    else:
        p += ".tif"
    if not os.path.exists(p):
        return None
    return tifffile.imread(p) if p.endswith(".tif") else cv2.imread(p, cv2.IMREAD_UNCHANGED)

def mask(shape, dx=0, dy=0):
    m = np.zeros(shape, bool)
    for l in boxes:
        x0, y0, x1, y1 = l["box"]
        a, b = int(x0 / SC - X0 + dx / SC), int(y0 / SC - Y0 + dy / SC)
        c, e = int(x1 / SC - X0 + dx / SC), int(y1 / SC - Y0 + dy / SC)
        a, b, c, e = max(a, 0), max(b, 0), min(c, shape[1]), min(e, shape[0])
        if c > a and e > b:
            m[b:e, a:c] = True
    return m

def auc(a, b, n=40000, seed=0):
    r = np.random.default_rng(seed)
    a = r.choice(a, min(n, a.size), replace=False); b = r.choice(b, min(n, b.size), replace=False)
    ranks = np.argsort(np.argsort(np.concatenate([a, b])))
    return (ranks[: a.size].sum() - a.size * (a.size - 1) / 2) / (a.size * b.size)

def sig(z, valid):
    zf = np.where(valid, z, 0).astype(np.float32)
    band = gaussian_filter(zf, 1.5) - gaussian_filter(zf, 12)
    gy, gx = np.gradient(gaussian_filter(zf, 2.0))
    jxx, jyy, jxy = (gaussian_filter(q, 25) for q in (gx * gx, gy * gy, gx * gy))
    coh = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy ** 2, 0)) / (jxx + jyy + 1e-6)
    return band[valid].std(), coh[valid].mean()

depths = sorted({int(os.path.basename(p).split("__d")[1].split("_")[0].split(".")[0]) for p in glob.glob(f"{ens}/{label}__seed42__d*")})
print("d  dir      " + "  ".join(f"{m:>8s}" for m in MODELS) + "   ENSEMBLE  | shifted-box AUC (ens) | ens band_std coh | n_models")
for d in depths:
    for rev in (False, True):
        zs, aucs, valid = {}, {}, None
        for m in MODELS:
            a = load(m, d, rev)
            if a is None:
                continue
            a = a.astype(np.float32)
            v = a > 0
            valid = v if valid is None else valid & v
            zs[m] = a
        if not zs:
            continue
        for m in zs:
            zs[m] = (zs[m] - zs[m][valid].mean()) / (zs[m][valid].std() + 1e-6)
        M = mask(valid.shape) & valid
        R = ~mask(valid.shape) & valid
        row = {m: auc(zs[m][M], zs[m][R]) for m in zs}
        e = np.mean([zs[m] for m in zs], axis=0)
        nulls = [auc(e[mask(valid.shape, dx, dy) & valid], e[~mask(valid.shape, dx, dy) & valid]) for dx, dy in ((-600, 0), (600, 0), (0, 300))]
        bs, ch = sig(e, valid)
        print(f"{d:+3d} {'reverse' if rev else 'forward':8s}" + "  ".join(f"{row.get(m, float('nan')):8.3f}" for m in MODELS)
              + f"   {auc(e[M], e[R]):8.3f}  | {nulls[0]:.3f} {nulls[1]:.3f} {nulls[2]:.3f} | {bs:5.2f} {ch:.3f} | {len(zs)}", flush=True)
