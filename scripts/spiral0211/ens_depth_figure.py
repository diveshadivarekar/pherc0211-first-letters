"""Figure: 4-model ensemble z-maps per depth (columns) for forward and reverse (rows), one PNG per label.

Usage: ens_depth_figure.py <ens_dir> <out.png> <label> [--depths -4,-2,0,2,4] [--box y0,y1,x0,x1] [--width 520] [--models seed42,seed43,dnative]
Display: z clipped to [-1, 3] (same scale for every panel and label), invalid = black, optional box outlined.
Run with D:/divesh-github/inkenv/Scripts/python.exe.
"""
import argparse, os
import numpy as np, tifffile, cv2
from scipy.ndimage import binary_erosion

ap = argparse.ArgumentParser()
ap.add_argument("ens"); ap.add_argument("out"); ap.add_argument("label")
ap.add_argument("--depths", default="-4,-2,0,2,4"); ap.add_argument("--box"); ap.add_argument("--width", type=int, default=520)
ap.add_argument("--models", default="seed42,seed43,dnative,hecate")
a = ap.parse_args()
MODELS = a.models.split(",")


def emap(d, rev):
    ms = []
    for m in MODELS:
        p = f"{a.ens}/{a.label}__{m}__d{d}" + ("_reverse" if rev else "") + ("_native.tif" if m == "hecate" else ".tif")
        if not os.path.exists(p):
            return None, None
        ms.append(tifffile.imread(p).astype(np.float32))
    valid = binary_erosion(np.all([v > 0 for v in ms], axis=0), iterations=30)
    z = [(v - v[valid].mean()) / (v[valid].std() + 1e-6) for v in ms]
    return np.mean(z, axis=0), valid


depths = [int(x) for x in a.depths.split(",")]
tiles = []
for rev in (False, True):
    row = []
    for d in depths:
        e, valid = emap(d, rev)
        if e is None:
            continue
        img = np.where(valid, np.clip((e + 1) / 4, 0, 1) * 255, 0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if a.box:
            y0, y1, x0, x1 = [int(v) for v in a.box.split(",")]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 8)
        h = int(img.shape[0] * a.width / img.shape[1])
        img = cv2.resize(img, (a.width, h), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"{'reverse' if rev else 'forward'} d={d:+d}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        row.append(np.pad(img, ((3, 3), (3, 3), (0, 0))))
    tiles.append(np.hstack(row))
cv2.imwrite(a.out, np.vstack(tiles))
print("wrote", a.out)
