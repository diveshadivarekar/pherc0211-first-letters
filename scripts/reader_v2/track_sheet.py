"""Track a papyrus sheet along the depth axis of a mesh-normal render and resample a stack centred on it.

The mesh render (66 layers, mesh at layer 33) is already sampled along the mesh normal, so a locally varying
offset d*(u,v) can be applied by re-sampling the layer axis. The sheet is located from the smoothed CT depth
profile at each pixel:
  --sheet A    : density peak nearest the mesh (M-6..M+4)
  --sheet B    : the density peak of the neighbouring sheet beyond the inter-sheet minimum (after the inter-sheet minimum, M+2..M+33)
  --edge       : with --sheet B, centre on B's near face (half-way between the minimum and B's peak) instead
Output: (2*half+1)-layer uint8 zarr (group with array "0"), centred layer `half`, same xy grid as the input render.
M = D//2 is the mesh layer of the input.

Usage: track_sheet.py <render66.zarr> <out.zarr> --sheet A|B [--edge] [--half 16]
Run with D:/divesh-github/inkenv/Scripts/python.exe.
"""
import argparse
import numpy as np, zarr
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter, map_coordinates, zoom

ap = argparse.ArgumentParser()
ap.add_argument("render"); ap.add_argument("out")
ap.add_argument("--sheet", choices=["A", "B"], required=True)
ap.add_argument("--edge", action="store_true")
ap.add_argument("--half", type=int, default=16)
a = ap.parse_args()

src = zarr.open(a.render, mode="r")["0"]
D, H, W = src.shape
M = D // 2                                   # mesh layer (33 for 66 layers, 50 for 100)
SUB = 4
small = np.asarray(src[:, ::SUB, ::SUB]).astype(np.float32)          # (D, h, w)
valid_s = small[D // 2] > 0
# smooth in xy (about 0.3 mm) and along depth, ignoring invalid pixels
num = np.stack([gaussian_filter(np.where(valid_s, s, 0), 8) for s in small])
den = gaussian_filter(valid_s.astype(np.float32), 8) + 1e-6
prof = gaussian_filter1d(num / den, 1.5, axis=0)                      # (D, h, w)

if a.sheet == "A":
    lo, hi = M - 6, M + 4
    k = lo + np.argmax(prof[lo:hi + 1], axis=0)
    center = k.astype(np.float32)
else:
    # inter-sheet minimum after the mesh sheet, then the next maximum
    lo_m, hi_m = M + 2, M + 13
    kmin = lo_m + np.argmin(prof[lo_m:hi_m + 1], axis=0)
    kpk = np.zeros_like(kmin)
    for i in range(kmin.shape[0]):
        for j in range(kmin.shape[1]):
            s, e = kmin[i, j], min(kmin[i, j] + 20, D - 1)
            kpk[i, j] = s + np.argmax(prof[s:e + 1, i, j])
    if a.edge:
        pmin = np.take_along_axis(prof, kmin[None], 0)[0]
        ppk = np.take_along_axis(prof, kpk[None], 0)[0]
        half = pmin + 0.5 * (ppk - pmin)
        center = np.zeros(kmin.shape, np.float32)
        for i in range(kmin.shape[0]):
            for j in range(kmin.shape[1]):
                seg = prof[kmin[i, j]:kpk[i, j] + 1, i, j]
                idx = np.nonzero(seg >= half[i, j])[0]
                center[i, j] = kmin[i, j] + (idx[0] if len(idx) else 0)
    else:
        center = kpk.astype(np.float32)

center = np.where(valid_s, center, np.nan)
fill = np.nanmedian(center)
center = np.where(np.isnan(center), fill, center)
center = gaussian_filter(median_filter(center, size=15), 4)           # robust, smooth in xy
center = np.clip(center, a.half + 1, D - 2 - a.half)
print(f"sheet {a.sheet}{' edge' if a.edge else ''}: layer median {np.median(center[valid_s]):.1f} "
      f"p5 {np.percentile(center[valid_s], 5):.1f} p95 {np.percentile(center[valid_s], 95):.1f}")

cen_full = zoom(center, (H / center.shape[0], W / center.shape[1]), order=1)[:H, :W]
if cen_full.shape != (H, W):
    cen_full = np.pad(cen_full, ((0, H - cen_full.shape[0]), (0, W - cen_full.shape[1])), mode="edge")

L = 2 * a.half + 1
grp = zarr.open_group(a.out, mode="w", zarr_format=2)
arr = grp.create_array("0", shape=(L, H, W), chunks=(L, 128, 128), dtype="uint8", fill_value=0)
BLK = 512
for x0 in range(0, W, BLK):
    x1 = min(W, x0 + BLK)
    vol = np.asarray(src[:, :, x0:x1]).astype(np.float32)             # (D, H, bw)
    c = cen_full[:, x0:x1]
    yy, xx = np.meshgrid(np.arange(H), np.arange(x1 - x0), indexing="ij")
    out = np.zeros((L, H, x1 - x0), np.uint8)
    for k in range(L):
        zc = c + (k - a.half)
        out[k] = np.clip(np.rint(map_coordinates(vol, [zc, yy, xx], order=1, mode="nearest")), 0, 255).astype(np.uint8)
    # keep the render's own validity mask (zero outside the mesh footprint)
    out[:, vol[D // 2] == 0] = 0
    arr[:, :, x0:x1] = out
np.save(a.out + ".center.npy", center.astype(np.float32))
print("wrote", a.out, arr.shape)
