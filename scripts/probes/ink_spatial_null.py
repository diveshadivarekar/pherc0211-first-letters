"""How much AUC does spatial structure alone produce? Slide the ink labels against an ink map and re-score.

The labels are a few large blobs and the model's output is smooth blobs, so two unrelated blob fields can correlate strongly, with either sign: there are tens of
independent blobs, not millions of independent pixels. This keeps the scored points where they are, re-assigns each point the label found ``(dr, dc)`` canvas pixels away
(wrapping inside the label window; only where that place is supervised), and repeats for random shifts of at least ``--min-shift`` pixels. The spread of those AUCs is the
null: how far from 0.5 an AUC can be with no real relation between the map and the labels at that position.

Usage (from the repo root):
    python scripts/probes/ink_spatial_null.py data/real_scans/paris4/w02 w02_20231031143852 data/processed/paris4_multi/w02/ink_s8/tile_ink.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from eval_ink_labels import label_points, score_from_pages  # noqa: E402


def auc(s: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 1000 or y.all() or not y.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)  # average ranks for ties
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    n1, n0 = int(y.sum()), int((~y).sum())  # python ints: numpy int32 overflows on Windows
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels_dir", type=Path)
    p.add_argument("prefix")
    p.add_argument("ink_h5", type=Path)
    p.add_argument("--tile-json", type=Path, default=None, help="default: <labels_dir>/tile.json")
    p.add_argument("--origin", type=int, nargs=3, default=None)
    p.add_argument("--shape", type=int, nargs=3, default=None)
    p.add_argument("--dmax", type=float, default=8.0)
    p.add_argument("--shifts", type=int, default=200)
    p.add_argument("--min-shift", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    if a.origin is None:
        t = json.loads((a.tile_json or a.labels_dir / "tile.json").read_text())
        origin, shape = np.array(t["origin_zyx"], float), np.array(t["shape_zyx"], float)
    else:
        origin, shape = np.array(a.origin, float), np.array(a.shape, float)
    lp = label_points(a.labels_dir, a.prefix, origin, shape)
    score = score_from_pages(a.ink_h5, lp, a.dmax)
    ok = np.isfinite(score)
    s, y0 = score[ok], lp["ink"][ok]
    rc = lp["rowcol"][ok]
    observed = auc(s, y0)
    h, w = lp["ink_img"].shape
    rng = np.random.default_rng(a.seed)
    nulls = []
    for _ in range(a.shifts):
        while True:
            dr, dc = int(rng.integers(-h, h)), int(rng.integers(-w, w))
            if max(abs(dr), abs(dc)) >= a.min_shift:
                break
        r, c = (rc[:, 0] + dr) % h, (rc[:, 1] + dc) % w
        sup = lp["sup_img"][r, c]
        if sup.sum() < 5000:
            continue
        nulls.append(auc(s[sup], lp["ink_img"][r, c][sup]))
    nulls = np.array([x for x in nulls if np.isfinite(x)])
    p_two = float((np.sum(np.abs(nulls - 0.5) >= abs(observed - 0.5)) + 1) / (len(nulls) + 1))
    res = {"labels": a.prefix, "ink_map": str(a.ink_h5), "points_scored": int(ok.sum()), "observed_auc": observed, "null_n": int(len(nulls)), "null_mean": float(nulls.mean()), "null_sd": float(nulls.std()),
           "null_p05": float(np.percentile(nulls, 5)), "null_p95": float(np.percentile(nulls, 95)), "null_min": float(nulls.min()), "null_max": float(nulls.max()), "p_two_sided": p_two}
    print(f"{a.prefix}: observed AUC {observed:.3f} on {int(ok.sum()):,} points | null over {len(nulls)} shifts: mean {nulls.mean():.3f}, sd {nulls.std():.3f}, 5-95% [{res['null_p05']:.3f}, {res['null_p95']:.3f}], range [{nulls.min():.3f}, {nulls.max():.3f}] | two-sided p = {p_two:.3f}")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
