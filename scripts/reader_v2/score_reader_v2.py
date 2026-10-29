"""Score Reader v2 outputs: label-free text screen (reader-v2's calibrated `textscore2.score2`) plus the
texture signature used elsewhere in this repo (coarse_std / band_std / coherence), in BOTH orientations.

For each <prefix>_d<D>.tif (forward) / <prefix>_d<D>_reverse.tif (reverse) pair:
  screen_fwd = score2(pf=forward, pr=reverse)   (reader-v2 convention: confident forward ink, not reverse)
  screen_rev = score2(pf=reverse, pr=forward)   (the winner's reading side was the reverse map)
Reader v2's own calibration: known text 0.010-0.032 (s180), blank/false 0.0003-0.0057, flag >= 0.008.
Because two orientations are tried, the max of the two is compared with that threshold only together with
the same two-orientation numbers on a blank/negative surface.

Usage: score_reader_v2.py <render.zarr> <out_csv> <prefix> [<prefix> ...]   (prefix = path without _d<D>.tif)
Run with D:/divesh-github/inkenv/Scripts/python.exe.
"""
import csv, glob, importlib.util, os, re, sys
import numpy as np, tifffile, zarr
from scipy.ndimage import gaussian_filter, binary_erosion

spec = importlib.util.spec_from_file_location("textscore2", "D:/divesh-github/reader-v2/scripts/textscore2.py")
ts = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts)

render, out_csv, prefixes = sys.argv[1], sys.argv[2], sys.argv[3:]
z = zarr.open(render, mode="r")
arr = z["0"] if "0" in z else z
ct = np.asarray(arr[arr.shape[0] // 2])


def sig(a, valid):
    v = binary_erosion(valid, iterations=30)
    if v.sum() < 5000:
        return float("nan"), float("nan"), float("nan")
    af = a.astype(np.float32)
    band = gaussian_filter(af, 1.5) - gaussian_filter(af, 12)
    gy, gx = np.gradient(gaussian_filter(af, 2.0))
    jxx, jyy, jxy = (gaussian_filter(q, 25) for q in (gx * gx, gy * gy, gx * gy))
    coh = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy ** 2, 0)) / (jxx + jyy + 1e-6)
    return float(af[v].std()), float(band[v].std()), float(coh[v].mean())


rows = []
for prefix in prefixes:
    for f in sorted(glob.glob(prefix + "_d*.tif")):
        if f.endswith("_reverse.tif"):
            continue
        r = f[:-4] + "_reverse.tif"
        if not os.path.exists(r):
            continue
        d = int(re.search(r"_d(-?\d+)\.tif$", f).group(1))
        pf, pr = tifffile.imread(f), tifffile.imread(r)
        h, w = pf.shape
        c = ct[:h, :w]
        valid = (c > 0) & (pf > 0)
        s_f = ts.score2(pf, pr, c, valid)
        s_r = ts.score2(pr, pf, c, valid)
        cf, bf, hf = sig(pf, valid)
        cr, br, hr = sig(pr, valid)
        rows.append(dict(map=os.path.basename(prefix), depth=d, valid_px=int(valid.sum()),
                         screen_fwd_180=s_f[180], screen_fwd_200=s_f[200], screen_rev_180=s_r[180], screen_rev_200=s_r[200],
                         coarse_fw=round(cf, 2), coarse_rv=round(cr, 2), band_fw=round(bf, 2), band_rv=round(br, 2),
                         coh_fw=round(hf, 3), coh_rv=round(hr, 3)))
        print(rows[-1], flush=True)
if rows:
    mode = "a" if os.path.exists(out_csv) else "w"
    with open(out_csv, mode, newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if mode == "w":
            wr.writeheader()
        wr.writerows(rows)
