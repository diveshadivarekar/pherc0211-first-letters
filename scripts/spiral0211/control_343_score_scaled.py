"""Positive control: does the ink_9um pipeline mark the winner's PHerc 343 letter boxes?

Crop frame: render crop (x 1500, y 600) of the raw 343 render; letter boxes are in raw render px.
Score = AUC (letter-box pixels vs the rest of the valid crop) and mean difference in z units, per depth
and direction, plus the same AUC after shifting the boxes by +-300/600 px (spatial null).
Run with D:/divesh-github/inkenv/Scripts/python.exe.
"""
import glob, json, re, sys
import numpy as np, tifffile

import os
SC = float(os.environ.get("SC", "1"))
X0, Y0 = 1500 / SC, 600 / SC
boxes = json.load(open("D:/divesh-github/virtual-unwrap/data/control_343/winner_files/letter_boxes.json"))["letters"]

def mask(shape, dx=0, dy=0):
    m = np.zeros(shape, bool)
    for l in boxes:
        x0, y0, x1, y1 = l["box"]
        a, b, c, d = int(x0 / SC - X0 + dx / SC), int(y0 / SC - Y0 + dy / SC), int(x1 / SC - X0 + dx / SC), int(y1 / SC - Y0 + dy / SC)
        a, b = max(a, 0), max(b, 0)
        c, d = min(c, shape[1]), min(d, shape[0])
        if c > a and d > b:
            m[b:d, a:c] = True
    return m

def auc(a, b, n=40000, seed=0):
    r = np.random.default_rng(seed)
    a = r.choice(a, min(n, a.size), replace=False); b = r.choice(b, min(n, b.size), replace=False)
    ranks = np.argsort(np.argsort(np.concatenate([a, b])))
    return (ranks[: a.size].sum() - a.size * (a.size - 1) / 2) / (a.size * b.size)

prefix = sys.argv[1]
print("depth dir      AUC(letters vs rest)  shifted-box AUCs (dx,dy: -600,0 | +600,0 | 0,+300)   mean z diff")
for f in sorted(glob.glob(prefix + "_d*.tif")):
    m = re.search(r"_d(-?\d+)(_reverse)?\.tif$", f)
    a = tifffile.imread(f).astype(np.float32)
    valid = a > 0
    z = (a - a[valid].mean()) / (a[valid].std() + 1e-6)
    M = mask(a.shape) & valid
    R = ~mask(a.shape) & valid
    s = [auc(z[mask(a.shape, dx, dy) & valid], z[~mask(a.shape, dx, dy) & valid]) for dx, dy in ((-600, 0), (600, 0), (0, 300))]
    print(f"{int(m.group(1)):+4d} {'reverse' if m.group(2) else 'forward':8s} {auc(z[M], z[R]):.3f}"
          f"   {s[0]:.3f} {s[1]:.3f} {s[2]:.3f}      {z[M].mean() - z[R].mean():+.2f}", flush=True)
