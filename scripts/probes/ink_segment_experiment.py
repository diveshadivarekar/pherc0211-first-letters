"""One labelled-segment tile end to end: does a wider refine search help, and is the model's depth sensitivity specific to one tile?

For ``data/real_scans/paris4/<segment>/`` (labels + geometry + ``tile.json`` from ``pick_label_tile.py`` + CT and surface-prediction crops):
builds the float tile, runs the split pipeline (M2 -> M3 -> fragment ink model, normal depth) twice - refine search +-8 (the default) and +-12 - and scores
both, and the segment's own refined surface ("gold"), on the label pixels all three reach. Then shifts the gold stack along the normal
(-6..+6 voxels) on the labels of the pipeline's biggest page. Writes ``<out>/<segment>.json``. Run from the repo root.

Usage:
    python scripts/probes/ink_segment_experiment.py w02 w02_20231031143852 --out data/processed/paris4_multi
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from eval_ink_labels import gold_scores, label_points, metrics, score_from_pages  # noqa: E402
from src.m2_segmentation.surface_segments import segment_from_surface_prediction  # noqa: E402
from src.m3_flattening.run import run as m3_run  # noqa: E402
from src.m4_ink_detection.run import run_fragment_model  # noqa: E402

MODEL = Path("data/processed/models/ink_unet_20k.pt")
UMBILICUS = Path("data/real_scans/paris4/umbilicus.json")


def auc(s: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if len(y) < 200 or y.all() or not y.any():
        return float("nan")
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    n1, n0 = int(y.sum()), int((~y).sum())  # python ints: numpy int32 overflows on Windows
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("segment", help="folder name under data/real_scans/paris4, e.g. w02")
    p.add_argument("prefix", help="file prefix of the labels, e.g. w02_20231031143852")
    p.add_argument("--out", type=Path, default=Path("data/processed/paris4_multi"))
    p.add_argument("--shifts", type=float, nargs="*", default=[-6, -3, 3, 6])
    p.add_argument("--dmax", type=float, default=8.0)
    a = p.parse_args()

    root = Path("data/real_scans/paris4") / a.segment
    out = a.out / a.segment
    out.mkdir(parents=True, exist_ok=True)
    tile_info = json.loads((root / "tile.json").read_text())
    origin, shape = tuple(tile_info["origin_zyx"]), tuple(tile_info["shape_zyx"])
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{a.segment} {time.time() - t0:5.0f}s] {msg}", flush=True)

    tile = root / "tile.h5"
    if not tile.exists():
        with h5py.File(root / "native_crop.h5", "r") as src, h5py.File(tile, "w") as dst:
            d = src["volume"]
            assert d.shape == shape and len(d.attrs["completed_slabs"]) == shape[0] // 128, "crop incomplete"
            ds = dst.create_dataset("volume", shape=d.shape, dtype=np.float32, chunks=(16, 256, 256))
            for z in range(0, d.shape[0], 64):
                ds[z : z + 64] = d[z : z + 64].astype(np.float32) / 255.0
            ds.attrs.update({"note": "uint8/255, M1 skipped", "voxel_size_um": 2.4, "origin_native_zyx": list(origin)})
        log("float tile built")

    seg = out / "seg.h5"
    s = segment_from_surface_prediction(root / "surface_pred_crop.h5", UMBILICUS, origin, shape, seg, wedge_degrees=40, min_voxels=200, split_wraps_apart=True)
    flat = m3_run(tile, seg, out / "m3", min_voxels=50000)
    log(f"split pipeline flattened ({s['n_segments']} pieces)")
    inks = {}
    for search in (8.0, 12.0):
        od = out / f"ink_s{int(search)}"
        od.mkdir(exist_ok=True)
        inks[search] = run_fragment_model(Path(flat), tile, od, MODEL, voxel_size_um=2.4, max_pages=200, flip_depth=False, refine=True, refine_search=search)
        log(f"ink maps, refine search +-{search:g} done")

    lp = label_points(root, a.prefix, np.array(origin, float), np.array(shape, float))
    ink = lp["ink"]
    log(f"{len(ink):,} supervised label pixels in the tile ({100 * ink.mean():.1f}% ink)")
    scores = {"pipeline +-8": score_from_pages(inks[8.0], lp, a.dmax), "pipeline +-12": score_from_pages(inks[12.0], lp, a.dmax)}
    scores["gold + refine"] = gold_scores(lp, tile, np.array(origin, float), MODEL, 2.4, False, refine=True)
    log("gold geometry scored")
    common = np.all([np.isfinite(v) for v in scores.values()], axis=0)
    result = {"segment": a.segment, "tile": tile_info, "label_pixels": int(len(ink)), "ink_fraction": float(ink.mean()), "dmax": a.dmax,
              "reached": {k: float(np.isfinite(v).mean()) for k, v in scores.items()},
              "common_pixels": int(common.sum()), "common_ink_fraction": float(ink[common].mean()) if common.any() else None,
              "common": {k: metrics(np.where(common, v, np.nan), ink) for k, v in scores.items()}}
    for k, m in result["common"].items():
        log(f"common subset ({int(common.sum()):,} px): {k:14s} AUC {m.get('auc', float('nan')):.3f}  F0.5 {m.get('f0.5', float('nan')):.3f}  (all-ink {m.get('f0.5_all_ink', float('nan')):.3f})")

    # depth scan of the official surface on the labels of the pipeline's biggest page
    tree_pts = {}
    best = (0, None)
    with h5py.File(inks[8.0], "r") as f:
        for k in f:
            cm = f[k]["coord_map"][...]
            v = ~np.isnan(cm[..., 0])
            d, _ = cKDTree(cm[v][::5]).query(lp["xyz_local"][::20], distance_upper_bound=a.dmax)
            n = int(np.isfinite(d).sum())
            if n > best[0]:
                best = (n, k)
        cm = f[best[1]]["coord_map"][...]
    v = ~np.isnan(cm[..., 0])
    d, _ = cKDTree(cm[v]).query(lp["xyz_local"], distance_upper_bound=a.dmax)
    select = np.isfinite(d) & common
    curve = {"0": auc(np.where(select, scores["gold + refine"], np.nan), ink)}
    for sh in a.shifts:
        g = gold_scores(lp, tile, np.array(origin, float), MODEL, 2.4, False, refine=True, depth_shift=sh, select=select)
        curve[f"{sh:+g}"] = auc(np.where(select, g, np.nan), ink)
        log(f"gold depth shift {sh:+g} voxels: AUC {curve[f'{sh:+g}']:.3f}")
    result["depth_scan_gold"] = {"page": best[1], "label_pixels": int(select.sum()), "ink_fraction": float(ink[select].mean()) if select.any() else None, "auc_by_shift": curve}
    (out / f"{a.segment}.json").write_text(json.dumps(result, indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
