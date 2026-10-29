"""Edge-band ink test: score only label points near a label edge, against the same spatial-shift null as ink_spatial_null.py.

Pooled AUC on Paris 4 is chance-level because the labels are a few large blobs and the model output is smooth blobs: any large-scale
trend (sheet brightness, depth drift, damage) decides the AUC. Close to a label edge, ink and no-ink pixels sit side by side, so such
trends cancel and only a detector that follows the label boundary scores well. Two statistics per band width B (canvas pixels):

* ``band``: AUC over supervised points within B pixels of an ink/no-ink edge (both sides).
* ``band_hp``: the same after high-passing the score map (score minus its Gaussian-weighted local mean, sigma = ``--hp-sigma``),
  which removes the remaining slow trend inside the band.

The null slides the whole label image (and therefore its edge band) by random shifts of at least ``--min-shift`` pixels and
recomputes both statistics, so observed and null are built the same way. Reports observed AUC, null 5-95 %, two-sided p.

Usage (from the repo root):
    python scripts/probes/ink_edge_band.py data/real_scans/paris4/w01 w01_20230702185753_r15 data/processed/paris4_multi/w01/ink_s8/tile_ink.h5 --out x.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from eval_ink_labels import label_points, score_from_pages  # noqa: E402
from probes.ink_spatial_null import auc  # noqa: E402


def band_image(ink: np.ndarray, sup: np.ndarray, b: float) -> np.ndarray:
    """Supervised pixels within ``b`` pixels of an ink/no-ink boundary (inside or outside the ink)."""
    d_in = ndi.distance_transform_edt(ink)  # inside ink: distance to the nearest no-ink pixel
    d_out = ndi.distance_transform_edt(~ink)  # outside: distance to the nearest ink pixel
    return sup & np.where(ink, d_in <= b, d_out <= b)


def highpass(rc: np.ndarray, s: np.ndarray, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Score minus its local mean (normalised Gaussian convolution over the scored pixels only)."""
    img, wt = np.zeros(shape, np.float64), np.zeros(shape, np.float64)
    img[rc[:, 0], rc[:, 1]] = s
    wt[rc[:, 0], rc[:, 1]] = 1.0
    num = ndi.gaussian_filter(img, sigma, mode="constant")
    den = ndi.gaussian_filter(wt, sigma, mode="constant")
    return s - (num / np.maximum(den, 1e-9))[rc[:, 0], rc[:, 1]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels_dir", type=Path)
    p.add_argument("prefix")
    p.add_argument("ink_h5", type=Path)
    p.add_argument("--tile-json", type=Path, default=None, help="default: <labels_dir>/tile.json")
    p.add_argument("--origin", type=int, nargs=3, default=None)
    p.add_argument("--shape", type=int, nargs=3, default=None)
    p.add_argument("--dmax", type=float, default=8.0)
    p.add_argument("--bands", type=float, nargs="+", default=[16, 32, 64])
    p.add_argument("--hp-sigma", type=float, default=48.0)
    p.add_argument("--shifts", type=int, default=200)
    p.add_argument("--min-shift", type=int, default=400)
    p.add_argument("--min-points", type=int, default=2000)
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
    s, rc = score[ok].astype(np.float64), lp["rowcol"][ok]
    ink_img, sup_img = lp["ink_img"], lp["sup_img"]
    h, w = ink_img.shape
    s_hp = highpass(rc, s, (h, w), a.hp_sigma)
    bands = {b: band_image(ink_img, sup_img, b) for b in a.bands}

    def stats(r: np.ndarray, c: np.ndarray) -> dict:
        out = {}
        sup = sup_img[r, c]
        y = ink_img[r, c]
        for b, bimg in bands.items():
            m = sup & bimg[r, c]
            if m.sum() < a.min_points:
                out[b] = (np.nan, np.nan, int(m.sum()))
                continue
            out[b] = (auc(s[m], y[m]), auc(s_hp[m], y[m]), int(m.sum()))
        return out

    obs = stats(rc[:, 0], rc[:, 1])
    rng = np.random.default_rng(a.seed)
    null = {b: ([], []) for b in a.bands}
    tries = 0
    while min(len(v[0]) for v in null.values()) < a.shifts and tries < a.shifts * 10:
        tries += 1
        while True:
            dr, dc = int(rng.integers(-h, h)), int(rng.integers(-w, w))
            if max(abs(dr), abs(dc)) >= a.min_shift:
                break
        st = stats((rc[:, 0] + dr) % h, (rc[:, 1] + dc) % w)
        for b, (x, xh, _) in st.items():
            if np.isfinite(x) and len(null[b][0]) < a.shifts:
                null[b][0].append(x)
                null[b][1].append(xh)

    res = {"labels": a.prefix, "ink_map": str(a.ink_h5), "points_scored": int(ok.sum()), "hp_sigma": a.hp_sigma, "min_shift": a.min_shift, "bands": {}}
    for b in a.bands:
        row = {"points": obs[b][2]}
        for name, k in (("band", 0), ("band_hp", 1)):
            nl = np.array(null[b][k])
            o = obs[b][k]
            if len(nl) < 10 or not np.isfinite(o):
                row[name] = {"observed": o, "null_n": int(len(nl))}
                continue
            pv = float((np.sum(np.abs(nl - 0.5) >= abs(o - 0.5)) + 1) / (len(nl) + 1))
            p_hi = float((np.sum(nl >= o) + 1) / (len(nl) + 1))  # one-sided: ink scores higher than no-ink
            row[name] = {"observed": o, "null_n": int(len(nl)), "null_mean": float(nl.mean()), "null_sd": float(nl.std()),
                         "null_p05": float(np.percentile(nl, 5)), "null_p95": float(np.percentile(nl, 95)), "p_two_sided": pv, "p_one_sided_high": p_hi}
            print(f"{a.prefix} B={b:g} {name:7s}: obs {o:.3f} on {obs[b][2]:,} pts | null n={len(nl)} mean {nl.mean():.3f} 5-95% "
                  f"[{np.percentile(nl, 5):.3f}, {np.percentile(nl, 95):.3f}] | p2 {pv:.3f} p_hi {p_hi:.3f}")
        res["bands"][str(b)] = row
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
