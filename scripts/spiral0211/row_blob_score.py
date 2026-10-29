"""Rows or blobs? Shape statistics of the 4-model ensemble z-map, per depth and direction.

Text on an unrolled sheet gives responses that line up along the writing direction (render x = along the winding) in
rows a few mm apart. A local artefact gives one compact clump. For each <label>, depth and direction this prints:
  agree   mean pairwise correlation of the 4 models (same as ensemble_agree.py)
  p99     99th percentile of the ensemble z
  hot%    % of valid area with smoothed ensemble z > 1.5
  ncl     number of hot clusters >= 0.1 mm^2 (1150 px at 9.36 um)
  top1%   share of the hot area inside the largest cluster (1 = one clump)
  acx/acy autocorrelation of the ensemble z at a 1 mm (107 px) shift along x vs along y; rows give acx > acy
  horiz%  share of hot-cluster area in clusters elongated along x (major/minor > 1.5, |angle| < 30 deg)
Optional --box y0,y1,x0,x1 adds the mean ensemble z inside the box minus the map mean (local depth profile).
Usage: row_blob_score.py <ens_dir> <label> [--models seed42,seed43,dnative] [--depths -4,-2,0,2,4] [--box y0,y1,x0,x1] [--exclude y0,y1,x0,x1] [--csv out.csv]
Run with D:/divesh-github/inkenv/Scripts/python.exe.
"""
import argparse, csv, glob, os
import numpy as np, tifffile
from scipy.ndimage import gaussian_filter, binary_erosion, label as cc_label, find_objects

ap = argparse.ArgumentParser()
ap.add_argument("ens"); ap.add_argument("label")
ap.add_argument("--depths"); ap.add_argument("--box"); ap.add_argument("--csv")
ap.add_argument("--exclude", help="y0,y1,x0,x1: drop this box (plus 100 px) from the valid area")
ap.add_argument("--models", default="seed42,seed43,dnative,hecate", help="Hecate is zero wherever the masked CT has holes (54-67%% of 0211 maps), so the 4-model mask keeps only 8-12%% of 0211; seed42,seed43,dnative keeps ~87%%")
a = ap.parse_args()
MODELS = a.models.split(",")
LAG = 107          # 1 mm at 9.362 um
MINCL = 1150       # 0.1 mm^2


def load(m, d, rev):
    p = f"{a.ens}/{a.label}__{m}__d{d}" + ("_reverse" if rev else "") + ("_native.tif" if m == "hecate" else ".tif")
    return tifffile.imread(p).astype(np.float32) if os.path.exists(p) else None


def ensemble(d, rev):
    maps = {m: load(m, d, rev) for m in MODELS}
    maps = {m: v for m, v in maps.items() if v is not None}
    if len(maps) < len(MODELS):
        return None
    valid = binary_erosion(np.all([v > 0 for v in maps.values()], axis=0), iterations=30)
    if a.exclude:
        y0, y1, x0, x1 = [int(v) for v in a.exclude.split(",")]
        valid[max(0, y0 - 100):y1 + 100, max(0, x0 - 100):x1 + 100] = False
    z = {m: (v - v[valid].mean()) / (v[valid].std() + 1e-6) for m, v in maps.items()}
    ms = list(z)
    rs = [np.corrcoef(gaussian_filter(z[ms[i]], 4)[valid][::7], gaussian_filter(z[ms[j]], 4)[valid][::7])[0, 1]
          for i in range(len(ms)) for j in range(i + 1, len(ms))]
    return np.mean(list(z.values()), axis=0), valid, float(np.mean(rs))


def autocorr(e, valid, dy, dx):
    x = np.where(valid, e - e[valid].mean(), 0)
    a1, a2 = x[dy:, dx:], x[:x.shape[0] - dy, :x.shape[1] - dx]
    v1, v2 = valid[dy:, dx:], valid[:valid.shape[0] - dy, :valid.shape[1] - dx]
    m = v1 & v2
    return float((a1[m] * a2[m]).mean() / (x[valid] ** 2).mean())


def shape_stats(e, valid):
    es = gaussian_filter(np.where(valid, e, 0), 4) / (gaussian_filter(valid.astype(np.float32), 4) + 1e-6)
    hot = (es > 1.5) & valid
    lab, n = cc_label(hot)
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.where(sizes >= MINCL)[0] + 1
    horiz = 0.0
    for i in keep:
        sl = find_objects((lab == i).astype(np.int8))[0]
        yy, xx = np.nonzero(lab[sl] == i)
        c = np.cov(np.stack([xx, yy]).astype(np.float64))
        ev, evec = np.linalg.eigh(c)
        elong = np.sqrt(ev[1] / max(ev[0], 1e-6))
        ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
        ang = (ang + 90) % 180 - 90
        if elong > 1.5 and abs(ang) < 30:
            horiz += sizes[i - 1]
    hot_area = sizes[keep - 1].sum() if len(keep) else 0
    return dict(hot=100 * hot.sum() / valid.sum(), ncl=len(keep), top1=(sizes[keep - 1].max() / hot_area) if hot_area else 0.0,
                horiz=100 * horiz / hot_area if hot_area else 0.0)


if a.depths:
    depths = [int(x) for x in a.depths.split(",")]
else:
    depths = sorted({int(os.path.basename(p).split("__d")[1].split("_")[0].split(".")[0]) for p in glob.glob(f"{a.ens}/{a.label}__seed42__d*")})
box = [int(v) for v in a.box.split(",")] if a.box else None
rows = []
print(f"{a.label}: d   dir      agree   p99   hot%  ncl  top1%  acx    acy    horiz%" + ("  box_dz" if box else ""))
for d in depths:
    for rev in (False, True):
        r = ensemble(d, rev)
        if r is None:
            continue
        e, valid, agree = r
        s = shape_stats(e, valid)
        row = dict(label=a.label, models=len(MODELS), d=d, dir="reverse" if rev else "forward", agree=agree, p99=float(np.percentile(e[valid], 99)),
                   hot=s["hot"], ncl=s["ncl"], top1=100 * s["top1"], acx=autocorr(e, valid, 0, LAG), acy=autocorr(e, valid, LAG, 0),
                   horiz=s["horiz"])
        if box:
            y0, y1, x0, x1 = box
            bv = valid[y0:y1, x0:x1]
            row["box_dz"] = float(e[y0:y1, x0:x1][bv].mean() - e[valid].mean()) if bv.sum() > 500 else float("nan")
        rows.append(row)
        print(f"{a.label}: {d:+3d} {row['dir']:8s} {agree:+.2f}  {row['p99']:5.2f} {row['hot']:5.1f} {row['ncl']:4d}  {row['top1']:5.1f}"
              f"  {row['acx']:+.3f} {row['acy']:+.3f}  {row['horiz']:5.1f}" + (f"  {row['box_dz']:+.2f}" if box else ""), flush=True)
if a.csv:
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "models", "d", "dir", "agree", "p99", "hot", "ncl", "top1", "acx", "acy", "horiz", "box_dz"])
        if new:
            w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
