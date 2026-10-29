"""Shift the sampled stack along the surface normal (default -9..+9 voxels) for the pipeline's page with the most labels and for the official surface, and score both on the same labels.

Run from the repo root (needs the Paris 4 tile, the pipeline output and scores_dmax8.npz; see docs/INK_EVALUATION.md). Optional args: the shifts."""
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from eval_ink_labels import gold_scores, label_points  # noqa: E402
from src.m4_ink_detection.scroll_pages import predict_page, sample_page_stack, surface_normals  # noqa: E402
from src.m4_ink_detection.train_fragments import load_model  # noqa: E402

ORIGIN = np.array([62080, 17280, 18816.0])
SHAPE = np.array([1536, 1408, 768.0])
TILE = Path("data/real_scans/paris4/tile.h5")
MODEL = Path("data/processed/models/ink_unet_20k.pt")
SHIFTS = [float(x) for x in sys.argv[1:]] or [-9, -6, -3, 0, 3, 6, 9]
lp = label_points(Path("data/real_scans/paris4/w00"), "w00_20231016151002", ORIGIN, SHAPE)
ink = lp["ink"]
npz = np.load("data/processed/paris4/scores_dmax8.npz")
gold0 = [npz[k] for k in npz.files if "gold" in k][0]


def auc(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if len(y) < 200 or y.all() or not y.any():
        return float("nan")
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# the split pipeline's page that holds most label points
ink_h5 = "data/processed/paris4/split/normal/tile_ink.h5"
best = (0, None)
with h5py.File(ink_h5, "r") as f:
    for k in f:
        cm = f[k]["coord_map"][...]
        v = ~np.isnan(cm[..., 0])
        d, _ = cKDTree(cm[v][::5]).query(lp["xyz_local"][::20], distance_upper_bound=8.0)
        n = int(np.isfinite(d).sum())
        if n > best[0]:
            best = (n, k)
    page = best[1]
    cm = f[page]["coord_map"][...]
print(f"pipeline page: {page}", flush=True)
v = ~np.isnan(cm[..., 0])
pts = cm[v]
d, idx = cKDTree(pts).query(lp["xyz_local"], distance_upper_bound=8.0)
on_page = np.isfinite(d)
select = on_page & np.isfinite(gold0)
print(f"{int(select.sum()):,} label pixels on this page that gold+refine also reaches (ink {ink[select].mean():.2f})", flush=True)
nrm = surface_normals(cm)
model, meta = load_model(MODEL)
rows = []
with h5py.File(TILE, "r") as tf:
    vol = tf["volume"]
    for sh in SHIFTS:
        t = time.time()
        cms = (cm + sh * np.nan_to_num(nrm)).astype(np.float32)
        cms[~v] = np.nan
        stack, valid = sample_page_stack(vol, cms, meta["n_slices"], 2.4, False)
        prob = predict_page(model, stack, valid, 2.4)
        sc = np.full(len(d), np.nan, np.float32)
        sc[on_page] = prob[v][idx[on_page]]
        pipe_auc = auc(np.where(select, sc, np.nan), ink)
        g = gold_scores(lp, TILE, ORIGIN, MODEL, 2.4, False, refine=True, depth_shift=sh, select=select)
        gold_auc = auc(np.where(select, g, np.nan), ink)
        rows.append((sh, pipe_auc, gold_auc))
        print(f"shift {sh:+5.1f} vox along the normal: pipeline page AUC {pipe_auc:.3f} | official surface AUC {gold_auc:.3f}  ({time.time() - t:.0f}s)", flush=True)
print("DEPTH SCAN DONE", flush=True)
