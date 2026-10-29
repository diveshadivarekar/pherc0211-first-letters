"""Re-refine one pipeline page (argv[1] = rank by label count, default 0) from its raw M3 coordinates with different search ranges and rescore on the same labels.

Run from the repo root (see docs/INK_EVALUATION.md). Not a tuning tool: AUC here swings by ~0.1 with a few voxels of depth."""
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from eval_ink_labels import label_points  # noqa: E402
from src.m4_ink_detection.scroll_pages import predict_page, refine_surface, sample_page_stack  # noqa: E402
from src.m4_ink_detection.train_fragments import load_model  # noqa: E402

ORIGIN = np.array([62080, 17280, 18816.0])
SHAPE = np.array([1536, 1408, 768.0])
TILE = Path("data/real_scans/paris4/tile.h5")
lp = label_points(Path("data/real_scans/paris4/w00"), "w00_20231016151002", ORIGIN, SHAPE)
ink = lp["ink"]
npz = np.load("data/processed/paris4/scores_dmax8.npz")
gold0 = [npz[k] for k in npz.files if "gold" in k][0]


def auc(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


import sys as _sys
_rank = int(_sys.argv[1]) if len(_sys.argv) > 1 else 0
_cands = []
with h5py.File("data/processed/paris4/split/normal/tile_ink.h5", "r") as _f:
    for _k in _f:
        _cm = _f[_k]["coord_map"][...]
        _v = ~np.isnan(_cm[..., 0])
        _d, _ = cKDTree(_cm[_v][::5]).query(lp["xyz_local"][::20], distance_upper_bound=8.0)
        _cands.append((int(np.isfinite(_d).sum()), _k))
_cands.sort(reverse=True)
PAGE = _cands[_rank][1]
print("page", PAGE, "(rank", _rank, "by labels)", flush=True)
with h5py.File("data/processed/paris4/split/normal/tile_ink.h5", "r") as f:
    cm_ref = f[PAGE]["coord_map"][...]            # refined once by the pipeline (what was scored)
with h5py.File("data/processed/paris4/split/m3/tile_flattened.h5", "r") as f:
    cm_raw = f[PAGE]["coord_map"][...]            # straight from M3, before refinement
v = ~np.isnan(cm_ref[..., 0])
d, idx = cKDTree(cm_ref[v]).query(lp["xyz_local"], distance_upper_bound=8.0)
on = np.isfinite(d)
select = on & np.isfinite(gold0)
model, meta = load_model(Path("data/processed/models/ink_unet_20k.pt"))
print(f"{int(select.sum()):,} label pixels (ink {ink[select].mean():.2f}); gold+refine AUC on them {auc(np.where(select, gold0, np.nan), ink):.3f}", flush=True)

with h5py.File(TILE, "r") as tf:
    vol = tf["volume"]

    def score(cm, label):
        t = time.time()
        # keep the label->pixel assignment fixed (pixel identity does not change when a surface is nudged); pixels refine drops are unscored
        stack, valid = sample_page_stack(vol, cm, meta["n_slices"], 2.4, False)
        prob = predict_page(model, stack, valid, 2.4)
        sc = np.full(len(d), np.nan, np.float32)
        sc[on] = prob[v][idx[on]]
        a = auc(np.where(select, sc, np.nan), ink)
        cov = float(np.isfinite(sc[select]).mean())
        print(f"{label:62s} AUC {a:.3f}  (reaches {100 * cov:.0f}% of the selected labels, {time.time() - t:.0f}s)", flush=True)

    score(cm_ref, "as scored: M3 page refined once (defaults, search +-8)")
    for search in (12, 16, 20):
        r, info = refine_surface(vol, cm_raw.copy(), search=float(search))
        score(r, f"raw M3 page, refine search +-{search} (mean shift {info['mean_abs_shift']:.1f} vox)")
print("REFINE VARIANTS DONE", flush=True)
