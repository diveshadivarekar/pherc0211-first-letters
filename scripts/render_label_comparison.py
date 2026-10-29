"""Render a page's CT, the model's ink probability and the ground-truth ink labels side by side, in page space.

For the page of a ``run_fragment_model`` output that overlaps the most supervised label pixels, every page pixel takes the label of the nearest
label point within ``--dmax`` voxels (see ``eval_ink_labels.py``); pixels with no label nearby are grey. Panels: CT | model ink probability | labels.

Usage:
    python scripts/render_label_comparison.py data/processed/paris4/split/normal/tile_ink.h5 --labels-dir data/real_scans/paris4/w00 \
        --prefix w00_20231016151002 --tile-origin 62080 17280 18816 --tile-shape 1536 1408 768 --out data/processed/paris4/figures/labels_split.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_ink_labels import label_points  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ink", type=Path)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--tile-origin", type=int, nargs=3, required=True)
    p.add_argument("--tile-shape", type=int, nargs=3, required=True)
    p.add_argument("--dmax", type=float, default=8.0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pages", type=int, default=1, help="render this many pages, most label-overlapped first")
    a = p.parse_args()

    lp = label_points(a.labels_dir, a.prefix, np.array(a.tile_origin, float), np.array(a.tile_shape, float))
    tree = cKDTree(lp["xyz_local"])
    ink = lp["ink"]
    best = []
    with h5py.File(a.ink, "r") as f:
        for k in f:
            cm = f[k]["coord_map"][...]
            v = ~np.isnan(cm[..., 0])
            d, i = tree.query(cm[v][::7], distance_upper_bound=a.dmax)
            best.append((int(np.isfinite(d).sum()), k))
        best.sort(reverse=True)
        for rank, (n_match, name) in enumerate(best[: a.pages], start=1):
            cm = f[name]["coord_map"][...]
            prob = f[name]["ink_probability"][...]
            ct = f[name]["page"][...]
            v = ~np.isnan(cm[..., 0])
            d, i = tree.query(cm[v], distance_upper_bound=a.dmax)
            gt = np.full(v.shape, -1, np.int8)
            ok = np.isfinite(d)
            tmp = np.full(v.sum(), -1, np.int8)
            tmp[ok] = ink[i[ok]].astype(np.int8)
            gt[v] = tmp
            lab = gt >= 0
            lo, hi = np.percentile(ct[v], [2, 99.5])
            g = np.clip((ct - lo) / (hi - lo + 1e-9), 0, 1)
            ctimg = np.stack([g, g, g], -1)
            ctimg[~v] = 0.1
            pimg = np.stack([np.clip(prob * 1.1, 0, 1), np.clip(prob, 0, 1) ** 1.5 * 0.7, np.clip(1 - prob, 0, 1) * 0.15], -1)
            pimg[~v] = 0.1
            gimg = np.full(v.shape + (3,), 0.1)
            gimg[v & ~lab] = [0.35, 0.35, 0.35]
            gimg[lab & (gt == 0)] = [0.12, 0.25, 0.45]
            gimg[lab & (gt == 1)] = [1.0, 0.85, 0.2]
            panels = [(x * 255).astype(np.uint8) for x in (ctimg, pimg, gimg)]
            gap = np.full((panels[0].shape[0], 8, 3), 255, np.uint8)
            row = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1)
            im = Image.fromarray(row)
            h, w = v.shape
            ImageDraw.Draw(im).text((6, 4), f"{name}   CT | model ink probability | ground truth (yellow = ink, blue = no ink, grey = unlabelled)   labelled {100 * lab.sum() / v.sum():.0f}% of page", fill=(255, 255, 255))
            out = a.out if a.pages == 1 else a.out.with_name(f"{a.out.stem}_{rank}{a.out.suffix}")
            out.parent.mkdir(parents=True, exist_ok=True)
            im.save(out)
            pm, ym = prob[lab], gt[lab] == 1
            print(f"{out}: {name}, {int(lab.sum()):,} labelled page pixels ({100 * ym.mean():.0f}% ink), mean model probability on ink {pm[ym].mean():.3f} vs on non-ink {pm[~ym].mean():.3f}")


if __name__ == "__main__":
    main()
