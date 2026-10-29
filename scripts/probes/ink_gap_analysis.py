"""Why do the pipeline pages score below the official segment's own refined surface? Orientation, distance and per-page breakdowns from saved ink maps.

Run from the repo root after `eval_ink_labels.py --scores-npz data/processed/paris4/scores_dmax8.npz` (see docs/INK_EVALUATION.md)."""
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from eval_ink_labels import label_points, metrics  # noqa: E402
from src.m2_segmentation.surface_segments import axis_at, load_umbilicus  # noqa: E402
from src.m4_ink_detection.scroll_pages import surface_normals  # noqa: E402

ORIGIN = np.array([62080, 17280, 18816.0])
SHAPE = np.array([1536, 1408, 768.0])
umb = load_umbilicus("data/real_scans/paris4/umbilicus.json")
lp = label_points(Path("data/real_scans/paris4/w00"), "w00_20231016151002", ORIGIN, SHAPE)
ink = lp["ink"]
npz = np.load("data/processed/paris4/scores_dmax8.npz")
gold = [npz[k] for k in npz.files if "gold" in k][0]
print(f"{len(ink):,} label points; gold+refine reaches {np.isfinite(gold).mean():.1%}")


def auc(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if len(y) < 50 or y.all() or not y.any():
        return float("nan")
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    n1, n0 = int(y.sum()), int((~y).sum())  # python ints: numpy's int32 overflows on Windows
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def radial_out(pts_abs):
    ay, ax = axis_at(umb, pts_abs[..., 0])
    r = np.stack([np.zeros_like(pts_abs[..., 0]), pts_abs[..., 1] - ay, pts_abs[..., 2] - ax], axis=-1)
    return r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-9)


# --- which way does the official surface's own normal point (relative to the scroll axis)? its 'normal' depth orientation is the one that works
g = lp["grid"]
gc = np.stack([g["z"], g["y"], g["x"]], axis=-1)
gv = lp["valid"]
gcm = np.where(gv[..., None], gc, np.nan).astype(np.float32)
gn = surface_normals(gcm)
inside = gv & ((gc >= ORIGIN) & (gc < ORIGIN + SHAPE)).all(-1)
ro = radial_out(gc)
dots = np.einsum("hwc,hwc->hw", np.nan_to_num(gn), ro)[inside]
gold_sign = float(np.sign(np.median(dots)))
print(f"official surface: median(normal . outward radial) = {np.median(dots):+.2f} -> sign {gold_sign:+.0f}; share of tile vertices with that sign {np.mean(np.sign(dots) == gold_sign):.0%}")


def load(path_normal, path_flip):
    pts, pn, pf, pid, sgn = [], [], [], [], []
    with h5py.File(path_normal, "r") as fn, h5py.File(path_flip, "r") as ff:
        for i, k in enumerate(fn):
            cm = fn[k]["coord_map"][...]
            assert np.allclose(np.nan_to_num(cm), np.nan_to_num(ff[k]["coord_map"][...])), "refined coord_maps differ between orientations"
            v = ~np.isnan(cm[..., 0])
            nrm = surface_normals(cm)
            ab = cm + ORIGIN
            d = np.einsum("hwc,hwc->hw", np.nan_to_num(nrm), radial_out(ab))
            pts.append(cm[v]); pn.append(fn[k]["ink_probability"][...][v]); pf.append(ff[k]["ink_probability"][...][v])
            pid.append(np.full(v.sum(), i)); sgn.append(np.full(v.sum(), np.sign(np.median(d[v]))))
    return np.concatenate(pts), np.concatenate(pn), np.concatenate(pf), np.concatenate(pid), np.concatenate(sgn), list(range(len(pn)))


for label, base in (("baseline", "data/processed/paris4/baseline"), ("split", "data/processed/paris4/split")):
    pts, pn, pf, pid, sgn, _ = load(f"{base}/normal/tile_ink.h5", f"{base}/flip/tile_ink.h5")
    d, i = cKDTree(pts).query(lp["xyz_local"], distance_upper_bound=8.0)
    ok = np.isfinite(d)
    sn = np.full(len(d), np.nan, np.float32); sf = sn.copy(); page = np.full(len(d), -1); psign = np.zeros(len(d))
    sn[ok], sf[ok], page[ok], psign[ok] = pn[i[ok]], pf[i[ok]], pid[i[ok]], sgn[i[ok]]
    common = ok & np.isfinite(gold)
    print(f"\n===== {label}: {int(ok.sum()):,} label points reached; {int(common.sum()):,} also reached by gold+refine")
    print(f"  distance label -> nearest page pixel (voxels): median {np.median(d[ok]):.1f}, p90 {np.percentile(d[ok], 90):.1f}")
    # orientation: global normal / flip, radially-consistent per page, and oracle per page
    oriented = np.where(psign == gold_sign, sn, sf)
    best = sn.copy()
    for pg in np.unique(page[page >= 0]):
        m = page == pg
        if auc(sf[m], ink[m]) > auc(sn[m], ink[m]):
            best[m] = sf[m]
    def on_common(s):
        return auc(np.where(common, s, np.nan), ink)
    print(f"  AUC on the common label pixels:  global normal {on_common(sn):.3f} | global flipped {on_common(sf):.3f} | radially-consistent per page {on_common(oriented):.3f} | oracle best per page {on_common(best):.3f} | gold+refine {on_common(gold):.3f}")
    print(f"  pages with outward-normal sign matching the official surface: {np.mean([np.mean(psign[page == p] == gold_sign) > 0.5 for p in np.unique(page[page >= 0])]):.0%} of pages touching labels")
    rows = []
    for pg in np.unique(page[page >= 0]):
        m = (page == pg) & common
        if m.sum() > 3000:
            rows.append((int(m.sum()), pg, float(ink[m].mean()), auc(sn[m], ink[m]), auc(sf[m], ink[m]), auc(gold[m], ink[m]), float(psign[m][0])))
    print("  per page (labelled px, page id, ink share, AUC normal, AUC flipped, AUC gold on the same px, outward sign):")
    for r in sorted(rows, reverse=True)[:8]:
        print(f"    {r[0]:8d}  page {r[1]:3d}  ink {r[2]:.2f}  normal {r[3]:.3f}  flipped {r[4]:.3f}  gold {r[5]:.3f}  sign {r[6]:+.0f}  {'<- matches official' if r[6] == gold_sign else ''}")
    # does the loss depend on how far the page is from the label?
    for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 8.01)):
        m = common & (d >= lo) & (d < hi)
        if m.sum() > 5000:
            print(f"  label->page distance {lo:g}-{hi:g} vox: {int(m.sum()):9,} px | AUC pipeline normal {auc(np.where(m, sn, np.nan), ink):.3f} vs gold on the same px {auc(np.where(m, gold, np.nan), ink):.3f}")
