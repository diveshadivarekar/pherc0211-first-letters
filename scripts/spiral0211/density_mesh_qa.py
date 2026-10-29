"""Density-profile mesh QA for a VC3D render (66 layers sampled along the mesh normal, mesh at the centre layer).

For every TxT tile the mean CT density per layer is taken over the valid pixels. A mesh that lies on a sheet shows
one bright sheet peak near the centre layer; holes, off-sheet patches, and meshes that straddle or jump between sheets
show no peak, an off-centre peak, or two comparable peaks. Raw density cannot find ink (see docs); this checks geometry.

Per tile: valid fraction at the centre layer, sheet-peak layer offset, peak contrast (grey levels, peak minus the
median profile outside +-W layers), FWHM in layers, and a second-peak ratio. A tile is "good" when it is mostly valid,
has contrast >= --min-contrast, one peak (second-peak ratio <= --max-second), and its peak lies within --offset-tol of the
winding's median peak offset (so a constant offset is reported, not penalised: use it as the ink depth shift).

Outputs in <out_prefix>: _tiles.csv, _maps.png (offset / contrast / good), _good_mask.tif (pixel mask, 255 = good),
and a printed summary. Thresholds default to values calibrated on the PHerc 343 control (see --help); compare windings
against the control run, not against absolute numbers.

Usage: density_mesh_qa.py <render.zarr> <out_prefix> [--tile 96] [--min-contrast 4] ...
Run with D:/divesh-github/inkenv/Scripts/python.exe (needs zarr, tifffile, cv2, scipy).
"""
import argparse, csv, json, os
import numpy as np, zarr, cv2, tifffile
from scipy.ndimage import gaussian_filter1d

ap = argparse.ArgumentParser()
ap.add_argument("render"); ap.add_argument("out_prefix")
ap.add_argument("--tile", type=int, default=96)
ap.add_argument("--center", type=int, default=None, help="mesh layer (default (n-1)//2)")
ap.add_argument("--search", type=int, default=14, help="peak search half-window around the centre layer")
ap.add_argument("--min-valid", type=float, default=0.8)
ap.add_argument("--min-contrast", type=float, default=6.0, help="grey levels (uint8)")
ap.add_argument("--offset-tol", type=int, default=3, help="layers around the winding's median peak offset")
ap.add_argument("--sigma", type=float, default=1.5, help="profile smoothing in layers")
ap.add_argument("--um", type=float, default=9.362)
a = ap.parse_args()

vol = zarr.open(a.render, mode="r")
arr = vol["0"] if "0" in vol else vol
n, H, W = arr.shape
c = a.center if a.center is not None else (n - 1) // 2
T = a.tile
ny, nx = H // T, W // T
lo, hi = max(c - a.search, 0), min(c + a.search, n - 1)
zz = np.arange(n)

rows = []
for ty in range(ny):
    strip = np.asarray(arr[:, ty * T:(ty + 1) * T, :nx * T])           # (n, T, nx*T) uint8
    v = (strip > 0).reshape(n, T, nx, T)
    s = strip.astype(np.float32).reshape(n, T, nx, T)
    cnt = v.sum(axis=(1, 3)).astype(np.float32)                         # (n, nx)
    prof = (s * v).sum(axis=(1, 3)) / np.maximum(cnt, 1)               # (n, nx)
    vfrac = cnt[c] / (T * T)
    for tx in range(nx):
        p = prof[:, tx].copy()
        ok = cnt[:, tx] > 0.2 * T * T
        rec = dict(ty=ty, tx=tx, x=tx * T + T // 2, y=ty * T + T // 2, valid=float(vfrac[tx]),
                   peak=np.nan, offset=np.nan, contrast=0.0, fwhm=np.nan, n_sheets=0, spacing=np.nan, mean=float(p[ok].mean()) if ok.any() else np.nan)
        if ok.sum() >= 0.8 * n:
            p[~ok] = np.interp(zz[~ok], zz[ok], p[ok])
            ps = gaussian_filter1d(p, a.sigma, mode="nearest")
            base = float(np.median(ps))
            pk = [i for i in range(2, n - 2) if ps[i] >= ps[i - 1] and ps[i] > ps[i + 1] and ps[i] - base > 0]
            if pk:
                prom_all = {i: ps[i] - base for i in pk}
                top = max(prom_all.values())
                keep = [i for i in pk if prom_all[i] >= max(0.5 * top, 1e-3)]
                z0 = min(keep, key=lambda i: (abs(i - c), -prom_all[i]))
                prom = float(prom_all[z0])
                half = base + 0.5 * prom
                l = z0
                while l > 0 and ps[l] > half: l -= 1
                r = z0
                while r < n - 1 and ps[r] > half: r += 1
                others = sorted(abs(i - z0) for i in keep if i != z0)
                spacing = float(others[0]) if others else np.nan
                rec.update(peak=z0, offset=z0 - c, contrast=prom, fwhm=r - l, n_sheets=len(keep), spacing=spacing)
        rows.append(rec)
    print(f"strip {ty + 1}/{ny}", end="\r", flush=True)
print()

ok_valid = np.array([r["valid"] >= a.min_valid and np.isfinite(r["offset"]) for r in rows])
off = np.array([r["offset"] for r in rows], float)
con = np.array([r["contrast"] for r in rows])
spc = np.array([r["spacing"] for r in rows], float)
strong = ok_valid & (con >= a.min_contrast)
# modal offset = centre of the --offset-tol window holding most strong tiles
cand_c = np.arange(-a.search - 6, a.search + 7)
cnt_c = [(np.abs(off[strong] - cc) <= a.offset_tol).sum() for cc in cand_c]
med_off = float(cand_c[int(np.argmax(cnt_c))]) if strong.any() else 0.0
good = strong & (np.abs(off - med_off) <= a.offset_tol)
for r, g in zip(rows, good):
    r["good"] = int(g)
# null: if the mesh sat at a random position relative to the sheets, offsets would be uniform within +-spacing/2.
rng = np.random.default_rng(0)
S = np.where(np.isfinite(spc[strong]), np.maximum(2 * spc[strong], 2 * a.offset_tol + 1), 2 * a.search + 1)
null_f = []
for _ in range(300):
    o = rng.uniform(-S / 2, S / 2)
    null_f.append(max((np.abs(o - cc) <= a.offset_tol).mean() for cc in cand_c))
null_best = float(np.mean(null_f)) if strong.any() else 0.0
obs_f = float(good.sum() / max(strong.sum(), 1))

with open(a.out_prefix + "_tiles.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

G = good.reshape(ny, nx)
mask = np.kron(G.astype(np.uint8) * 255, np.ones((T, T), np.uint8))
full = np.zeros((H, W), np.uint8); full[:mask.shape[0], :mask.shape[1]] = mask
tifffile.imwrite(a.out_prefix + "_good_mask.tif", full, compression="zlib")

def heat(vals, vmin, vmax):
    img = np.clip((np.nan_to_num(vals, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
    return cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
O = off.reshape(ny, nx); C = con.reshape(ny, nx)
panels = [heat(O, med_off - 12, med_off + 12), heat(C, 0, max(12.0, np.nanpercentile(con, 98))), cv2.cvtColor((G * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]
scale = max(1, 1200 // nx)
panels = [cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) for p in panels]
cv2.imwrite(a.out_prefix + "_maps.png", np.vstack(panels))

fw = np.array([r["fwhm"] for r in rows], float)
excess = (obs_f - null_best) / max(1 - null_best, 1e-6)
summ = dict(render=a.render, shape=[n, H, W], tiles=int(len(rows)), valid_tiles=int(ok_valid.sum()),
            hole_tiles_pct=float(100 * (~(np.array([r["valid"] for r in rows]) >= a.min_valid)).mean()),
            strong_pct_of_valid=float(100 * strong.sum() / max(ok_valid.sum(), 1)),
            modal_offset_layers=med_off,
            on_sheet_pct_of_strong=100 * obs_f, random_placement_expect_pct=100 * null_best, excess_concentration=float(excess),
            good_pct_of_all=float(100 * good.sum() / len(rows)),
            median_contrast=float(np.median(con[ok_valid])) if ok_valid.any() else None,
            median_sheet_spacing_layers=float(np.nanmedian(spc[strong])) if strong.any() else None,
            median_fwhm_um=float(np.nanmedian(fw[strong]) * a.um) if strong.any() else None)
print(json.dumps(summ, indent=1))
json.dump(summ, open(a.out_prefix + "_summary.json", "w"), indent=1)
