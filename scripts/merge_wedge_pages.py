"""Merge wedge pages that continue the same sheet across a wedge seam, and render the merged page and a scroll overview.

M2 cuts every sheet into angular wedges, so one sheet of the scroll ends up as several pages. Two pages are joined when their
pixels meet across a wedge seam (a multiple of ``--wedge`` degrees about the scroll axis) at the same height and radius -
touching *different* wraps sit at different radii and are not joined. Writes, next to ``--out``:
  <out>_merged_page.png   the largest merged group unrolled onto an (angle, z) canvas at ~1 voxel/pixel: CT left, ink overlay right
  <out>_overview.png      every group on the (angle, z) map, one colour per merged group (grey = a page that stayed alone)
  <out>_groups.json       pages per group, pixels, angular span, median radius

The unrolling is a cylindrical projection about the axis, not an isometric flattening: it is a view, not a reading surface.

Usage:
    python scripts/merge_wedge_pages.py data/processed/ink_split/normal/pherc0332_native_tile_ink.h5         --out data/processed/ink_split/figures/split --origin 16384 7680 7680 --umbilicus data/real_scans/pherc0332_umbilicus.json
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.m2_segmentation.surface_segments import axis_at, load_umbilicus  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("ink", help="ink .h5 from run_fragment_model (groups with coord_map, page, ink_probability)")
ap.add_argument("--out", required=True, help="output prefix")
ap.add_argument("--origin", type=float, nargs=3, metavar=("Z", "Y", "X"), required=True, help="tile origin in scroll voxels (the coord_maps are tile-local)")
ap.add_argument("--umbilicus", required=True)
ap.add_argument("--wedge", type=float, default=40.0, help="wedge size M2 used (degrees)")
args = ap.parse_args()
ink_path, prefix, WEDGE = args.ink, args.out, args.wedge
ORIGIN = np.array(args.origin, float)
umb = load_umbilicus(args.umbilicus)

pages = []
with h5py.File(ink_path, "r") as f:
    for k in sorted(f, key=lambda n: int(n.split("_")[1])):
        cm = f[k]["coord_map"][...]
        v = ~np.isnan(cm[..., 0])
        p = cm[v].astype(np.float64) + ORIGIN
        ay, ax = axis_at(umb, p[:, 0])
        th = np.degrees(np.arctan2(p[:, 1] - ay, p[:, 2] - ax)) % 360
        r = np.hypot(p[:, 1] - ay, p[:, 2] - ax)
        pages.append({"name": k, "z": p[:, 0], "th": th, "r": r, "ink": f[k]["ink_probability"][...][v], "ct": f[k]["page"][...][v]})
n = len(pages)
print(f"{n} pages, {sum(len(p['z']) for p in pages):,} pixels")

# --- join pages across wedge seams: same height z, same radius r, facing pixels within 1.2 degrees of the seam
parent = list(range(n))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


pair_counts = {}
for b in np.arange(0, 360, WEDGE):
    lo_pts, lo_id, up_pts, up_id = [], [], [], []
    for i, p in enumerate(pages):
        c = ((p["th"] - b + 180) % 360) - 180
        lo, up = (c >= -1.2) & (c < 0), (c >= 0) & (c < 1.2)
        if lo.any():
            lo_pts.append(np.stack([p["z"][lo], p["r"][lo]], 1)); lo_id.append(np.full(lo.sum(), i))
        if up.any():
            up_pts.append(np.stack([p["z"][up], p["r"][up]], 1)); up_id.append(np.full(up.sum(), i))
    if not lo_pts or not up_pts:
        continue
    lo_pts, lo_id, up_pts, up_id = np.concatenate(lo_pts), np.concatenate(lo_id), np.concatenate(up_pts), np.concatenate(up_id)
    d, j = cKDTree(up_pts).query(lo_pts, distance_upper_bound=5.0)
    ok = np.isfinite(d) & (up_id[np.minimum(j, len(up_id) - 1)] != lo_id)
    for a, bb in zip(lo_id[ok], up_id[j[ok]]):
        pair_counts[(int(a), int(bb))] = pair_counts.get((int(a), int(bb)), 0) + 1
joined = 0
for (a, bb), c in pair_counts.items():
    if c >= 30:
        ra, rb = find(a), find(bb)
        if ra != rb:
            parent[ra] = rb; joined += 1
groups = {}
for i in range(n):
    groups.setdefault(find(i), []).append(i)
glist = sorted(groups.values(), key=lambda g: -sum(len(pages[i]["z"]) for i in g))


def rel_angle(th, centre):
    return ((th - centre + 180) % 360) - 180


stats = []
for g in glist:
    th = np.concatenate([pages[i]["th"] for i in g]); centre = np.degrees(np.arctan2(np.sin(np.radians(th)).mean(), np.cos(np.radians(th)).mean()))
    rel = rel_angle(th, centre)
    stats.append({"pages": [pages[i]["name"] for i in g], "n_pages": len(g), "pixels": int(sum(len(pages[i]["z"]) for i in g)), "angular_span_deg": float(np.percentile(rel, 99.5) - np.percentile(rel, 0.5)), "median_r": float(np.median(np.concatenate([pages[i]["r"] for i in g]))), "centre_deg": float(centre)})
print(f"{joined} seam joins -> {len(glist)} groups (from {n} pages); largest groups:")
for s in stats[:6]:
    print(f"  {s['n_pages']:2d} pages, {s['pixels']:>9,} px, span {s['angular_span_deg']:5.1f} deg, median radius {s['median_r']:5.0f} vox")
multi = [s for s in stats if s["n_pages"] > 1]
print(f"groups made of >1 page: {len(multi)}; pages in them: {sum(s['n_pages'] for s in multi)}/{n}; largest angular span {max(s['angular_span_deg'] for s in stats):.0f} deg")
json.dump({"pages": n, "seam_joins": joined, "groups": len(glist), "stats": stats}, open(prefix + "_groups.json", "w"), indent=1)


def splat(g, centre, r_med, width=None, z0=None, z1=None):
    """Unroll a group onto an (angle, z) canvas at ~1 voxel/pixel at the group's median radius."""
    pxs = []
    for i in g:
        p = pages[i]
        pxs.append((rel_angle(p["th"], centre), p["z"], p["ink"], p["ct"], p["r"]))
    ang = np.concatenate([a for a, *_ in pxs]); zz = np.concatenate([b for _, b, *_ in pxs]); ink = np.concatenate([c for *_, c, _, _ in pxs]); ct = np.concatenate([d for *_, d, _ in pxs])
    scale = r_med * np.pi / 180  # pixels per degree
    col = np.round((ang - ang.min()) * scale).astype(int); row = np.round(zz - zz.min()).astype(int)
    h, w = row.max() + 1, col.max() + 1
    cnt = np.zeros((h, w)); np.add.at(cnt, (row, col), 1)
    ctm = np.zeros((h, w)); np.add.at(ctm, (row, col), ct)
    inkm = np.zeros((h, w)); np.maximum.at(inkm, (row, col), ink)
    cov = cnt > 0
    ctm = np.divide(ctm, cnt, out=np.zeros_like(ctm), where=cov)
    # close pinholes between splats (the page grid is not aligned with the canvas): fill from covered neighbours
    wgt = ndi.uniform_filter(cov.astype(float), 3); fill = (~cov) & (wgt > 0.35)
    for arr in (ctm, inkm):
        sm = np.divide(ndi.uniform_filter(arr * cov, 3), wgt, out=np.zeros_like(arr), where=wgt > 0)
        arr[fill] = sm[fill]
    cov = cov | fill
    return ctm, inkm, cov, scale, ang.min()


def to_img(ctm, inkm, cov, overlay):
    lo, hi = np.percentile(ctm[cov], [2, 99.5])
    g = np.clip((ctm - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = np.stack([g, g, g], -1)
    if overlay:
        a = np.clip((inkm - 0.15) / 0.6, 0, 1)[..., None]
        hot = np.stack([np.ones_like(inkm), np.clip(inkm * 1.2, 0, 1) * 0.55, np.zeros_like(inkm)], -1)
        rgb = rgb * (1 - 0.85 * a) + hot * (0.85 * a)
    rgb[~cov] = [0.12, 0.12, 0.14]
    return (rgb * 255).astype(np.uint8)


# ---- the merged page: the largest group, CT on the left and with ink on the right
g = glist[0]; s0 = stats[0]
ctm, inkm, cov, scale, a0 = splat(g, s0["centre_deg"], s0["median_r"])
left, right = to_img(ctm, inkm, cov, False), to_img(ctm, inkm, cov, True)
canvas = np.concatenate([left, np.full((left.shape[0], 8, 3), 255, np.uint8), right], axis=1)
img = Image.fromarray(canvas)
d = ImageDraw.Draw(img)
# mark the wedge seams that were stitched
w = left.shape[1]
for b in np.arange(0, 360, WEDGE):
    rel = ((b - s0["centre_deg"] + 180) % 360) - 180
    x = (rel - a0) * scale
    if 0 < x < w:
        for off in (0, w + 8):
            d.line([(x + off, 0), (x + off, 14)], fill=(80, 200, 255), width=2)
img.save(prefix + "_merged_page.png")
print(f"merged page {canvas.shape[1] // 2}x{canvas.shape[0]} px per panel, {s0['n_pages']} pages, ink pixels > 0.5: {100 * (inkm[cov] > 0.5).mean():.2f}%")

# ---- overview: every group on the (angle, z) map, one colour per group; inner wraps drawn first
zmin = min(p["z"].min() for p in pages); zmax = max(p["z"].max() for p in pages)
H, W = int(np.ceil(zmax - zmin)) + 2, 720
ov = np.full((H, W, 3), 22, np.uint8)
rng = np.random.default_rng(3)
for k, gg in sorted(enumerate(glist), key=lambda kv: stats[kv[0]]["median_r"]):
    colour = (np.array(rng.uniform(70, 255, 3))).astype(np.uint8) if stats[k]["n_pages"] > 1 else np.array([110, 110, 120], np.uint8)
    for i in gg:
        p = pages[i]
        r_ = np.minimum(np.round(p["z"] - zmin).astype(int), H - 1); c_ = np.minimum((p["th"] * 2).astype(int), W - 1)
        ov[r_, c_] = colour
ov = ndi.maximum_filter(ov, size=(2, 2, 1))
ovi = Image.fromarray(ov)
ImageDraw.Draw(ovi).text((6, 4), f"{len(glist)} groups from {n} pages; colour = merged group (grey = single page); x = angle 0-360 deg, y = z", fill=(255, 255, 255))
ovi.save(prefix + "_overview.png")
print("wrote", prefix + "_merged_page.png", prefix + "_overview.png")
