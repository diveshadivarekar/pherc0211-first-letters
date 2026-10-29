"""Does splitting touching wraps (M2 ``split_wraps``) give better flattened pages? Measured with the tifxyz QA.

For one tile of the native scroll crop and each configuration - the plain M2 pieces, then the pieces split at
one or more link tolerances - this segments the tile from the surface prediction, flattens it (M3), exports
tifxyz, audits it with ``src.tifxyz`` and prints one comparison table. Everything is recomputed from the crop.
``--ablate`` adds, at the first tolerance, the split without the single-valued rule and the cylindrical
("sheet") page builder, to show which part of the pipeline does the work.

Usage (the tile the tolerance was tuned on, then a tile disjoint from it):
    python scripts/eval_wrap_split.py --name tuned --origin 128 512 512 --shape 768 1024 1024 --tol 3 2 --ablate
    python scripts/eval_wrap_split.py --name heldout --origin 128 0 0 --shape 768 1024 512 --tol 3 2 --ablate --min-voxels 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import export_tifxyz  # noqa: E402
import scan_tiles  # noqa: E402
import tifxyz_qa  # noqa: E402

from src import tifxyz as tx  # noqa: E402
from src.m2_segmentation.surface_segments import load_umbilicus, segment_from_surface_prediction  # noqa: E402
from src.m3_flattening.run import run as m3_run  # noqa: E402


def evaluate_config(tile_h5: Path, abs_origin: tuple, shape: tuple, out: Path, label: str, split: dict | bool, method: str, min_voxels: int, umb, pred, wedge: float = 40.0) -> dict:
    seg = out / "seg.h5"
    segment_from_surface_prediction(scan_tiles.SURFACE, scan_tiles.UMBILICUS, abs_origin, shape, seg, wedge_degrees=wedge, min_voxels=200, split_wraps_apart=split)
    flat = m3_run(tile_h5, seg, out / "m3", min_voxels=min_voxels, method=method, umbilicus=scan_tiles.UMBILICUS if method == "sheet" else None)
    index = export_tifxyz.export(Path(flat), out / "tifxyz", abs_origin, stride=4, voxel_um=scan_tiles.VOXEL_UM, min_valid=1000, overwrite=True)
    args = argparse.Namespace(threshold=127.0, search=12.0, max_angle=60.0)
    rows = []
    for item in index:
        r = tifxyz_qa.audit_one(Path(item["path"]), args, umb, pred, None, None, None)
        g = r["geometry"]
        rows.append((g["valid_vertices"], r["radial_extent"]["wraps_crossed"], r["support"]["on_surface_fraction"], r["radiality"].get("cross_wrap_fraction", np.nan), g.get("cell_shear_cos_p95", np.nan), r["radial_extent"].get("wraps_crossed_reliable", True)))
    v, w, s, c, sh, rel = (np.array(x, dtype=float) for x in zip(*rows))
    rel = rel.astype(bool)
    ok = ~np.isnan(c)
    return {
        "config": label, "pages": len(v), "vertices": int(v.sum()), "pages_ge_10k_vertices": int((v >= 10000).sum()),
        "median_page_vertices": int(np.median(v)), "area_on_pages_over_2_wraps_pct": float(100 * v[(w > 2) & rel].sum() / v.sum()),  # only pages short enough for the measure to mean anything
        "pages_too_long_for_wrap_count": int((~rel).sum()), "largest_page_vertices": int(v.max()),
        "wraps_per_slice_weighted": float(np.average(w, weights=v)), "max_wraps_per_slice": float(w.max()),
        "on_surface_weighted": float(np.average(s, weights=v)), "cross_wrap_weighted": float((c[ok] * v[ok]).sum() / v[ok].sum()),
        "shear_cos_p95_weighted": float(np.average(sh, weights=v)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="output folder name under --out")
    p.add_argument("--origin", type=int, nargs=3, metavar=("Z", "Y", "X"), required=True, help="tile corner inside the native crop")
    p.add_argument("--shape", type=int, nargs=3, metavar=("Z", "Y", "X"), default=(768, 1024, 1024))
    p.add_argument("--tol", type=float, nargs="*", default=[3.0], help="split link tolerances in native voxels (a baseline row is always included)")
    p.add_argument("--wedge", type=float, default=40.0, help="angular cut (degrees) of M2 pieces; bigger = larger pages but the plane page builder degrades past ~80-120")
    p.add_argument("--ablate", action="store_true", help="also run the first tolerance without the single-valued rule and with the sheet page builder")
    p.add_argument("--min-voxels", type=int, default=100000, help="M3 skips segments smaller than this")
    p.add_argument("--out", type=Path, default=Path("data/processed/eval_wrap_split"))
    a = p.parse_args()

    root = a.out / a.name
    root.mkdir(parents=True, exist_ok=True)
    tile_h5 = root / "tile.h5"
    scan_tiles.cut_tile(tuple(a.origin), tuple(a.shape), tile_h5)
    abs_origin = tuple(int(scan_tiles.CROP_ORIGIN[i] + a.origin[i]) for i in range(3))
    umb, pred = load_umbilicus(scan_tiles.UMBILICUS), tx.load_volume(scan_tiles.SURFACE)
    results = []
    configs = [("plain M2 + plane", False, "auto")] + [(f"split tol {t:g} + plane", {"link_tol": float(t)}, "auto") for t in a.tol]
    if a.ablate and a.tol:
        t = a.tol[0]
        configs += [(f"split tol {t:g} (multi-valued) + plane", {"link_tol": float(t), "single_valued": False}, "auto"), (f"split tol {t:g} + sheet", {"link_tol": float(t)}, "sheet")]
    results = []
    for i, (label, split, method) in enumerate(configs):
        cfg = root / f"cfg{i}"
        cfg.mkdir(exist_ok=True)
        res = evaluate_config(tile_h5, abs_origin, tuple(a.shape), cfg, label, split, method, a.min_voxels, umb, pred, a.wedge)
        results.append(res)
        print(f"{res['config']:34s} pages {res['pages']:3d} | vtx {res['vertices']:7d} | >=10k {res['pages_ge_10k_vertices']:2d} | largest {res['largest_page_vertices']:6d} | multi-wrap area {res['area_on_pages_over_2_wraps_pct']:5.1f}% | on-surface {res['on_surface_weighted']:.3f} | cross-wrap {res['cross_wrap_weighted']:.3f} | shear {res['shear_cos_p95_weighted']:.2f}", flush=True)
    (root / "results.json").write_text(json.dumps({"tile_origin_native": abs_origin, "shape": list(a.shape), "wedge_degrees": a.wedge, "results": results}, indent=2))
    tile_h5.unlink()  # derived float32 copy of the tile (GBs)


if __name__ == "__main__":
    main()
