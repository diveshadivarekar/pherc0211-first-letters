"""Score ink-detection output against official ink labels given in a segment's own (tifxyz) surface.

The Vesuvius ink labels are 2D binary images aligned to a traced segment; the segment's ``x/y/z.tif`` grid (coarse: one cell per
1/scale canvas pixels) gives every label pixel a 3D position in the scroll volume. This places the supervised label pixels that
fall inside a tile into 3D and scores an ink map (a ``run_fragment_model`` output whose pages carry a refined ``coord_map``) at those
points: each label point takes the ink probability of the nearest page pixel within ``--dmax`` voxels, so any flattening can be
scored on the same labels, and the share of labels a flattening reaches is reported as its coverage.

``--gold`` also runs the model on the *segment's own geometry* (the page the labels were drawn on): that separates a model that
does not transfer to this scan from pages that are bad.

Usage:
    python scripts/eval_ink_labels.py --labels-dir data/real_scans/paris4/w00 --prefix w00_20231016151002 \
        --tile-origin 62080 17280 18816 --tile-shape 1536 1408 768 \
        --ink baseline=data/processed/paris4/baseline/normal/..._ink.h5 split=data/processed/paris4/split/normal/..._ink.h5 \
        --gold data/real_scans/paris4/tile.h5 --model data/processed/models/ink_unet_20k.pt --out data/processed/paris4/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.m4_ink_detection.infer import best_fbeta  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def read_image(path: Path, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    im = Image.open(path)
    if box is not None:
        im = im.crop(box)  # (left, top, right, bottom)
    return np.asarray(im)


def label_points(labels_dir: Path, prefix: str, origin: np.ndarray, shape: np.ndarray, margin: float = 0.0) -> dict:
    """Supervised label pixels whose 3D position lies inside the tile: tile-local ``(z, y, x)``, ink flag, canvas row/col."""
    meta = json.loads((labels_dir / "meta.json").read_text())
    sx, sy = meta["scale"]
    g = {k: read_image(labels_dir / f"{k}.tif").astype(np.float64) for k in "xyz"}
    valid = (g["z"] > 0) & np.isfinite(g["z"])
    lo, hi = origin - margin, origin + shape + margin
    inside = valid & (g["z"] >= lo[0]) & (g["z"] < hi[0]) & (g["y"] >= lo[1]) & (g["y"] < hi[1]) & (g["x"] >= lo[2]) & (g["x"] < hi[2])
    if not inside.any():
        raise SystemExit("no part of the labelled segment lies inside this tile")
    rows, cols = np.nonzero(inside)
    r0, r1 = int(rows.min() / sy), int(rows.max() / sy) + 2
    c0, c1 = int(cols.min() / sx), int(cols.max() / sx) + 2
    ink = read_image(labels_dir / f"{prefix}_inklabels.tif", (c0, r0, c1, r1)) > 0
    sup = read_image(labels_dir / f"{prefix}_supervision_mask.tif", (c0, r0, c1, r1)) > 0
    rr, cc = np.mgrid[r0 : r0 + ink.shape[0], c0 : c0 + ink.shape[1]]
    coords = np.stack([rr * sy, cc * sx])  # grid coordinates of every canvas pixel
    pts = {k: ndi.map_coordinates(g[k], coords, order=1, mode="nearest") for k in "zyx"}
    ok_corner = ndi.map_coordinates(valid.astype(np.float64), coords, order=1, mode="nearest") > 0.999  # all four grid corners valid
    p = np.stack([pts["z"], pts["y"], pts["x"]], axis=-1)
    keep = sup & ok_corner & ((p >= origin) & (p < origin + shape)).all(axis=-1)
    return {
        "xyz_local": (p[keep] - origin).astype(np.float32), "ink": ink[keep], "window": (r0, c0, r1, c1),
        "rowcol": np.stack([rr[keep] - r0, cc[keep] - c0], axis=-1), "ink_img": ink, "sup_img": sup, "grid": g, "meta": meta, "valid": valid,
        "n_sup_window": int(sup.sum()),
    }


def metrics(score: np.ndarray, ink: np.ndarray) -> dict:
    """Coverage (share of label points with a score), AUC, best F0.5 over thresholds, and the trivial 'everything is ink' F0.5."""
    cov = np.isfinite(score)
    s, y = score[cov], ink[cov]
    out = {"label_points": int(len(score)), "covered": int(cov.sum()), "coverage": float(cov.mean()), "ink_prevalence": float(y.mean()) if len(y) else float("nan")}
    if len(y) == 0 or y.all() or not y.any():
        return out
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # ties get average ranks
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    avg = (np.cumsum(cnt) - (cnt - 1) / 2)
    ranks = avg[inv]
    n1, n0 = int(y.sum()), int((~y).sum())
    out["auc"] = float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    f = best_fbeta(s.astype(np.float32).clip(0, 1), y, np.ones(len(s), bool))
    out.update({"f0.5": f["f_beta"], "precision": f["precision"], "recall": f["recall"], "threshold": f["threshold"]})
    p = out["ink_prevalence"]
    out["f0.5_all_ink"] = float(1.25 * p / (0.25 * p + 1))
    return out


def score_from_pages(ink_h5: Path, lp: dict, dmax: float) -> np.ndarray:
    """Ink probability at each label point from the nearest valid page pixel of a ``run_fragment_model`` output."""
    pts, probs = [], []
    with h5py.File(ink_h5, "r") as f:
        for k in f:
            cm = f[k]["coord_map"][...]
            v = ~np.isnan(cm[..., 0])
            pts.append(cm[v]); probs.append(f[k]["ink_probability"][...][v])
    pts, probs = np.concatenate(pts), np.concatenate(probs)
    d, i = cKDTree(pts).query(lp["xyz_local"], distance_upper_bound=dmax)
    score = np.full(len(d), np.nan, dtype=np.float32)
    ok = np.isfinite(d)
    score[ok] = probs[i[ok]]
    return score


def gold_scores(lp: dict, tile_h5: Path, origin: np.ndarray, model_path: Path, voxel_um: float, flip: bool, refine: bool = False, depth_shift: float = 0.0, select: np.ndarray | None = None) -> np.ndarray:
    """Run the model on the labelled segment's own surface (the geometry the labels were drawn on).

    The segment re-enters the tile at several, possibly distant, canvas locations, so each connected island of label pixels is
    processed on its own bounding box. ``refine`` snaps each page onto the papyrus first (``scroll_pages.refine_surface``), as
    the pipeline does for its own pages - labels drawn on an older scan can sit several voxels off the sheet at this resolution.
    ``depth_shift`` moves the sampled stack that many voxels along the surface normal (after refining), to probe how sensitive the
    model is to where the ink surface sits in its window; ``select`` (bool per label point) limits the work to islands holding selected points.
    """
    from src.m4_ink_detection.scroll_pages import predict_page, refine_surface, sample_page_stack, surface_normals
    from src.m4_ink_detection.train_fragments import load_model

    r0, c0, r1, c1 = lp["window"]
    sx, sy = lp["meta"]["scale"]
    model, meta = load_model(model_path)
    scores = np.full(len(lp["ink"]), np.nan, dtype=np.float32)
    h, w = lp["ink_img"].shape
    kept = np.zeros((h, w), bool)
    kept[lp["rowcol"][:, 0], lp["rowcol"][:, 1]] = True
    islands, n = ndi.label(ndi.binary_dilation(kept, iterations=6), structure=np.ones((3, 3)))
    with h5py.File(tile_h5, "r") as f:
        vol = f["volume"]
        tile_shape = np.array(vol.shape)
        for k, sl in enumerate(ndi.find_objects(islands), start=1):
            ys, xs = sl
            sel = (lp["rowcol"][:, 0] >= ys.start) & (lp["rowcol"][:, 0] < ys.stop) & (lp["rowcol"][:, 1] >= xs.start) & (lp["rowcol"][:, 1] < xs.stop)
            if sel.sum() < 500 or (select is not None and (sel & select).sum() < 500):
                continue
            rr, cc = np.mgrid[ys.start : ys.stop, xs.start : xs.stop]
            coords = np.stack([(rr + r0) * sy, (cc + c0) * sx])
            cm = np.stack([ndi.map_coordinates(lp["grid"][c], coords, order=1, mode="nearest") for c in "zyx"], axis=-1) - origin
            ok = ndi.map_coordinates(lp["valid"].astype(np.float64), coords, order=1, mode="nearest") > 0.999
            cm[~(ok & ((cm >= 0) & (cm < tile_shape)).all(axis=-1))] = np.nan
            cm = cm.astype(np.float32)
            if refine:
                cm, _ = refine_surface(vol, cm)
            if depth_shift:
                cm = (cm + depth_shift * np.nan_to_num(surface_normals(cm))).astype(np.float32)  # NaN (off-page) pixels stay NaN
            stack, valid = sample_page_stack(vol, cm, meta["n_slices"], voxel_um, flip)
            prob = predict_page(model, stack, valid, voxel_um)
            rows_, cols_ = lp["rowcol"][sel, 0] - ys.start, lp["rowcol"][sel, 1] - xs.start
            got = valid[rows_, cols_]
            idx = np.flatnonzero(sel)[got]
            scores[idx] = prob[rows_[got], cols_[got]]
    return scores


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--tile-origin", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    p.add_argument("--tile-shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    p.add_argument("--ink", nargs="*", default=[], help="name=path pairs of run_fragment_model outputs (one per flattening and depth orientation)")
    p.add_argument("--dmax", type=float, default=4.0, help="max distance (voxels) from a label point to the page pixel that scores it")
    p.add_argument("--gold", type=Path, default=None, help="float tile .h5 (volume): also score the model on the segment's own geometry")
    p.add_argument("--gold-variants", nargs="*", default=["raw-normal", "raw-flip", "refine-normal", "refine-flip"], choices=["raw-normal", "raw-flip", "refine-normal", "refine-flip"], help="which gold-geometry runs to do (each takes minutes)")
    p.add_argument("--scores-npz", type=Path, default=None, help="save every method's per-label-point scores here (for re-analysis without recomputing)")
    p.add_argument("--model", type=Path, default=Path("data/processed/models/ink_unet_20k.pt"))
    p.add_argument("--voxel-um", type=float, default=2.4)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    origin, shape = np.array(a.tile_origin, float), np.array(a.tile_shape, float)
    lp = label_points(a.labels_dir, a.prefix, origin, shape)
    print(f"{len(lp['ink']):,} supervised label pixels inside the tile ({100 * lp['ink'].mean():.1f}% ink); canvas window {lp['window']}", flush=True)
    results, scores = {}, {}
    if a.gold:
        for variant in a.gold_variants:
            refine, flip = variant.startswith("refine"), variant.endswith("flip")
            name = "gold geometry" + (" + refine" if refine else "") + ", " + ("flipped" if flip else "normal")
            scores[name] = gold_scores(lp, a.gold, origin, a.model, a.voxel_um, flip, refine)
            results[name] = metrics(scores[name], lp["ink"])
            print(f"{name:52s} coverage {100 * results[name]['coverage']:5.1f}% | AUC {results[name].get('auc', float('nan')):.3f} | F0.5 {results[name].get('f0.5', float('nan')):.3f} (all-ink {results[name].get('f0.5_all_ink', float('nan')):.3f})", flush=True)
    for item in a.ink:
        name, path = item.split("=", 1)
        scores[name] = score_from_pages(Path(path), lp, a.dmax)
        results[name] = metrics(scores[name], lp["ink"])
    common = {}
    normal = [n for n in scores if "flip" not in n]
    if len(normal) > 1:  # every method scored on exactly the label pixels that ALL of them reach
        mask = np.all([np.isfinite(scores[n]) for n in normal], axis=0)
        common = {n: metrics(np.where(mask, scores[n], np.nan), lp["ink"]) for n in normal}
        print()
        print(f"Common label pixels reached by all of: {', '.join(normal)}  ({int(mask.sum()):,} = {100 * mask.mean():.1f}% of supervised)")
        for n, m in common.items():
            print(f"  {n:50s} AUC {m.get('auc', float('nan')):.3f} | F0.5 {m.get('f0.5', float('nan')):.3f} (all-ink {m.get('f0.5_all_ink', float('nan')):.3f}) | precision {m.get('precision', float('nan')):.2f} recall {m.get('recall', float('nan')):.2f}")
        print()
    if a.scores_npz:
        a.scores_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.scores_npz, ink=lp["ink"], xyz_local=lp["xyz_local"], **{f"s{i}_{n}": v for i, (n, v) in enumerate(scores.items())})
    for name, m in results.items():
        print(f"{name:52s} coverage {100 * m['coverage']:5.1f}% | AUC {m.get('auc', float('nan')):.3f} | F0.5 {m.get('f0.5', float('nan')):.3f} (all-ink {m.get('f0.5_all_ink', float('nan')):.3f}) | precision {m.get('precision', float('nan')):.2f} recall {m.get('recall', float('nan')):.2f}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"tile_origin": a.tile_origin, "tile_shape": a.tile_shape, "dmax": a.dmax, "label_pixels": int(len(lp["ink"])), "ink_fraction": float(lp["ink"].mean()), "results": results, "common_subset": common}, indent=1))


if __name__ == "__main__":
    main()
