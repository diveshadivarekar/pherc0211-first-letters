"""Snap a mesh-normal render onto one papyrus sheet (a stand-in for the winner's unpublished snap_meshes.py).

The 66-layer VC3D render is already sampled along the mesh normal (mesh at layer M = D//2), so a smooth per-pixel depth
offset z*(u,v) can be applied by resampling the layer axis instead of re-rendering. The sheet is found from the CT depth
profile (lightly smoothed in xy so neighbouring sheets, 15-17 layers apart, stay separate):

1. Every local maximum of the depth profile within M +- --search layers is a candidate, with its prominence
   (peak minus the profile minimum within +-8 layers).
2. Region growing keeps ONE sheet: seeded at the most prominent candidate, each neighbouring pixel takes the candidate
   closest to its parent's depth if it is within --max-step layers (so the surface cannot hop to the next sheet,
   which is ~16 layers away), and no candidate further than --max-dev from the mesh is used; the frontier is expanded strongest-first. Unreached pixels get new seeds only if
   they are at least --min-patch pixels; each patch is reported (patches may sit on different sheets).
3. Unassigned pixels are filled from the nearest assigned one and flagged; the depth map is smoothed and upsampled.

Output: <out.zarr> (group, array "0", same D x H x W as the input, sheet at layer M, edge layers padded with the nearest
source layer; pixels never assigned are zero), <out>.depth.npy (coarse depth map, layers), <out>.json (stats).
Usage: snap_render.py <render.zarr> <out.zarr> [--sub 4] [--search 12] [--max-step 2] [--max-dev 8]
Run with D:/divesh-github/inkenv/Scripts/python.exe.
"""
import argparse, heapq, json
import numpy as np, zarr
from scipy.ndimage import gaussian_filter, gaussian_filter1d, distance_transform_edt, label, map_coordinates, zoom, minimum_filter1d

ap = argparse.ArgumentParser()
ap.add_argument("render"); ap.add_argument("out")
ap.add_argument("--sub", type=int, default=4, help="coarse grid step in render pixels")
ap.add_argument("--xy-sigma", type=float, default=2.0, help="xy smoothing in coarse pixels")
ap.add_argument("--search", type=int, default=12)
ap.add_argument("--max-step", type=float, default=2.0, help="max depth change between neighbouring coarse pixels (layers)")
ap.add_argument("--min-prom", type=float, default=2.0, help="min candidate prominence (grey levels)")
ap.add_argument("--min-patch", type=int, default=2000, help="min patch size in coarse pixels")
ap.add_argument("--max-dev", type=float, default=8.0, help="max distance from the fitted mesh (layers); about half a sheet spacing, so the snap cannot reach the next sheet")
ap.add_argument("--smooth", type=float, default=1.5, help="final depth smoothing in coarse pixels")
a = ap.parse_args()

src = zarr.open(a.render, mode="r")
src = src["0"] if "0" in src else src
D, H, W = src.shape
M = D // 2
S = a.sub
small = np.asarray(src[:, ::S, ::S]).astype(np.float32)
valid = small[M] > 0
num = np.stack([gaussian_filter(np.where(valid, s, 0), a.xy_sigma) for s in small])
den = gaussian_filter(valid.astype(np.float32), a.xy_sigma) + 1e-6
prof = gaussian_filter1d(num / den, 1.0, axis=0)
prom_base = minimum_filter1d(prof, 17, axis=0)
lo, hi = M - a.search, M + a.search
k = np.arange(lo, hi + 1)
ismax = (prof[k] >= prof[k - 1]) & (prof[k] > prof[k + 1])
prom = np.where(ismax, prof[k] - prom_base[k], -np.inf)          # (K, h, w)
# sub-layer peak position by parabolic fit
pm, p0, pp = prof[k - 1], prof[k], prof[k + 1]
dd = pm - 2 * p0 + pp
frac = np.where(np.abs(dd) > 1e-6, 0.5 * (pm - pp) / np.where(np.abs(dd) > 1e-6, dd, 1), 0)
zpos = k[:, None, None] + np.clip(frac, -0.5, 0.5)
ok_c = ismax & (prom >= a.min_prom) & valid[None] & (np.abs(zpos - M) <= a.max_dev)
h, w = valid.shape

depth = np.full((h, w), np.nan, np.float32)
patch = np.zeros((h, w), np.int32)
cand_z = [[None] * w for _ in range(h)]
ys, xs = np.nonzero(ok_c.any(0))
for y, x in zip(ys, xs):
    sel = ok_c[:, y, x]
    cand_z[y][x] = (zpos[sel, y, x], prom[sel, y, x])

def grow(seed_y, seed_x, seed_z, pid):
    heap = [(0.0, seed_y, seed_x, seed_z)]
    n = 0
    while heap:
        negp, y, x, z = heapq.heappop(heap)
        if not np.isnan(depth[y, x]):
            continue
        depth[y, x] = z; patch[y, x] = pid; n += 1
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and np.isnan(depth[yy, xx]) and cand_z[yy][xx] is not None:
                zc, pc = cand_z[yy][xx]
                j = int(np.argmin(np.abs(zc - z)))
                if abs(zc[j] - z) <= a.max_step:
                    heapq.heappush(heap, (-float(pc[j]), yy, xx, float(zc[j])))
    return n

# seed order: strongest candidate nearest the mesh layer first
best = np.where(ok_c, prom - 0.5 * np.abs(zpos - M), -np.inf).max(0)
order = np.argsort(-best, axis=None)
patches = []
pid = 0
for idx in order:
    y, x = divmod(int(idx), w)
    if not np.isfinite(best[y, x]):
        break
    if not np.isnan(depth[y, x]) or cand_z[y][x] is None:
        continue
    zc, pc = cand_z[y][x]
    j = int(np.argmax(pc - 0.5 * np.abs(zc - M)))
    pid += 1
    n = grow(y, x, float(zc[j]), pid)
    if n < a.min_patch:                       # too small: undo and block this seed
        m = patch == pid
        depth[m] = np.nan; patch[m] = 0; pid -= 1
        cand_z[y][x] = None
        continue
    patches.append(dict(id=pid, coarse_px=int(n), median_offset=float(np.nanmedian(depth[patch == pid]) - M)))

assigned = ~np.isnan(depth)
frac_assigned = float(assigned[valid].mean()) if valid.any() else 0.0
if assigned.any():
    _, (iy, ix) = distance_transform_edt(~assigned, return_indices=True)
    filled = depth[iy, ix]
else:
    filled = np.full((h, w), float(M), np.float32)
filled = gaussian_filter(filled, a.smooth)
off = filled[assigned & valid] - M
jump = np.abs(np.diff(np.where(assigned, depth, np.nan), axis=1))
jump_rate = float(np.nanmean(jump > 4)) if np.isfinite(jump).any() else float("nan")
stats = dict(render=a.render, shape=[D, H, W], coarse=[h, w], sub=S, assigned_pct_of_valid=100 * frac_assigned, patches=patches,
             offset_median=float(np.median(off)) if off.size else None, offset_p5=float(np.percentile(off, 5)) if off.size else None,
             offset_p95=float(np.percentile(off, 95)) if off.size else None, neighbour_jump_gt4_rate=jump_rate, params=vars(a))
print(json.dumps({k: v for k, v in stats.items() if k != "params"}, indent=1))

cen = zoom(filled, (H / h, W / w), order=1)
cen = np.pad(cen, ((0, max(0, H - cen.shape[0])), (0, max(0, W - cen.shape[1]))), mode="edge")[:H, :W]
amask = zoom(assigned.astype(np.uint8), (H / h, W / w), order=0)
amask = np.pad(amask, ((0, max(0, H - amask.shape[0])), (0, max(0, W - amask.shape[1]))), mode="edge")[:H, :W].astype(bool)

grp = zarr.open_group(a.out, mode="w", zarr_format=2)
arr = grp.create_array("0", shape=(D, H, W), chunks=(D, 128, 128), dtype="uint8", fill_value=0)
BLK = 512
for x0 in range(0, W, BLK):
    x1 = min(W, x0 + BLK)
    vol = np.asarray(src[:, :, x0:x1]).astype(np.float32)
    c = cen[:, x0:x1]
    yy, xx = np.meshgrid(np.arange(H), np.arange(x1 - x0), indexing="ij")
    out = np.empty((D, H, x1 - x0), np.uint8)
    for kk in range(D):
        zc = np.clip(c + (kk - M), 0, D - 1)
        out[kk] = np.clip(np.rint(map_coordinates(vol, [zc, yy, xx], order=1, mode="nearest")), 0, 255).astype(np.uint8)
    bad = (vol[M] == 0) | ~amask[:, x0:x1]
    out[:, bad] = 0
    arr[:, :, x0:x1] = out
    print(f"cols {x1}/{W}", end="\r", flush=True)
print()
np.save(a.out + ".depth.npy", np.where(assigned, depth, np.nan).astype(np.float32))
json.dump(stats, open(a.out + ".json", "w"), indent=1)
print("wrote", a.out)
