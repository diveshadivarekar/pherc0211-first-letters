"""Scan tiles of the native-resolution scroll crop for candidate ink regions.

For each tile: cut it from the crop, segment it from the official surface prediction, flatten the
sheets (M3), run the fragment-trained ink model on the refined pages in both depth orientations
(which side of a page is "up" is not known from geometry), and flag candidate regions with their
position in the scan. Outputs per tile: ``candidates.json`` and a PNG gallery per orientation.

Everything here is a *candidate for a human to inspect*. The model has not been validated on this scroll.

Usage:
    python scripts/scan_tiles.py --out data/processed/scan
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.m2_segmentation.surface_segments import segment_from_surface_prediction  # noqa: E402
from src.m3_flattening.run import run as m3_run  # noqa: E402
from src.m4_ink_detection.candidates import find_candidates, to_scroll_coordinates  # noqa: E402
from src.m4_ink_detection.run import run_fragment_model  # noqa: E402
from src.imaging import to_display_image  # noqa: E402

CROP = Path("data/real_scans/pherc0332_native_crop.h5")
SURFACE = Path("data/real_scans/pherc0332_surface_pred_crop.h5")
UMBILICUS = Path("data/real_scans/pherc0332_umbilicus.json")
MODEL = Path("data/processed/models/ink_unet_20k.pt")
CROP_ORIGIN = (16256, 7168, 7168)  # native-voxel corner of the crop
VOXEL_UM = 2.399


def cut_tile(origin_in_crop: tuple[int, int, int], shape: tuple[int, int, int], out: Path) -> None:
    z0, y0, x0 = origin_in_crop
    with h5py.File(CROP, "r") as src, h5py.File(out, "w") as f:
        ds = f.create_dataset("volume", shape=shape, dtype=np.float32, chunks=(16, shape[1], shape[2]))
        for z in range(0, shape[0], 64):
            ds[z : z + 64] = src["volume"][z0 + z : z0 + z + 64, y0 : y0 + shape[1], x0 : x0 + shape[2]].astype(np.float32) / 255.0


def panel_for_gallery(arr: np.ndarray, hot: bool, size: int) -> Image.Image:
    """Square thumbnail of ``arr`` (aspect preserved, white padding), percentile-stretched."""
    img = to_display_image(arr, cmap="hot" if hot else None).convert("RGB")
    img.thumbnail((size, size))
    tile = Image.new("RGB", (size, size), "white")
    tile.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return tile


def gallery(candidates: list[dict], flattened: Path, ink: Path, out: Path, max_items: int = 12, cell: int = 220) -> None:
    """Contact sheet: for each candidate, the page and the ink map side by side, region boxed."""
    if not candidates:
        return
    items = candidates[:max_items]
    cols = 2
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (2 * cell + 12), rows * (cell + 26)), "white")
    d = ImageDraw.Draw(sheet)
    with h5py.File(ink, "r") as fi:
        for i, c in enumerate(items):
            g = fi[c["page"]]
            page, prob = g["page"][...], g["ink_probability"][...]
            x0, y0, x1, y1 = c["bbox"]
            pad = 60
            wy0, wy1, wx0, wx1 = max(0, y0 - pad), min(page.shape[0], y1 + pad), max(0, x0 - pad), min(page.shape[1], x1 + pad)
            tiles = [panel_for_gallery(page[wy0:wy1, wx0:wx1], hot=False, size=cell), panel_for_gallery(prob[wy0:wy1, wx0:wx1], hot=True, size=cell)]
            ox, oy = (i % cols) * (2 * cell + 12), (i // cols) * (cell + 26)
            for k, t in enumerate(tiles):
                sheet.paste(t, (ox + k * cell, oy + 20))
            d.text((ox + 2, oy + 3), f"{c['page']}  p={c['mean_prob']:.2f} area={c['area_px']} elong={c['elongation']:.1f}  zyx={[int(v) for v in c['centre_native_zyx']]}", fill="#1f2933")
    sheet.save(out)


def scan_tile(name: str, origin_in_crop: tuple[int, int, int], shape: tuple[int, int, int], out_root: Path, reuse: bool = False, split_wraps: bool = False, wedge_degrees: float = 40.0) -> dict:
    t0 = time.time()
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    tile_h5 = out / f"{name}_corrected.h5"
    abs_origin = tuple(int(CROP_ORIGIN[a] + origin_in_crop[a]) for a in range(3))
    seg = out / f"{name}_segmented.h5"
    flat = out / f"{name}_flattened.h5"
    reusing = reuse and flat.exists() and all((out / f"ink_{o}" / f"{name}_ink.h5").exists() for o in ("normal", "flip"))
    if reusing:
        print(f"[{name}] re-using the saved pages and ink maps; recomputing candidates only", flush=True)
        s = {"n_segments": -1}
    else:
        print(f"[{name}] cutting tile {origin_in_crop} {shape}", flush=True)
        cut_tile(origin_in_crop, shape, tile_h5)
        s = segment_from_surface_prediction(SURFACE, UMBILICUS, abs_origin, shape, seg, wedge_degrees=wedge_degrees, min_voxels=200, split_wraps_apart=split_wraps)
        print(f"[{name}] {s['n_segments']} surface pieces", flush=True)
        flat = m3_run(tile_h5, seg, out, min_voxels=100000)

    result = {"tile": name, "origin_native_zyx": list(abs_origin), "shape": list(shape), "n_segments": s["n_segments"], "orientations": {}}
    for orient in ("normal", "flip"):
        odir = out / f"ink_{orient}"
        odir.mkdir(exist_ok=True)
        ink = odir / f"{name}_ink.h5"
        if not reusing:
            ink = run_fragment_model(flat, tile_h5, odir, MODEL, voxel_size_um=VOXEL_UM, max_pages=200, flip_depth=orient == "flip", refine=True)
        cands = []
        with h5py.File(ink, "r") as fi:
            for page in fi:
                for c in find_candidates(fi[page]["ink_probability"][...], fi[page]["coord_map"][...]):
                    cands.append({"page": page, **to_scroll_coordinates(c, abs_origin, VOXEL_UM)})
        cands.sort(key=lambda c: -c["score"])
        gallery(cands, flat, ink, out / f"gallery_{orient}.png")
        result["orientations"][orient] = {"n_candidates": len(cands), "candidates": cands}
        print(f"[{name}] {orient}: {len(cands)} candidate regions", flush=True)

    if tile_h5.exists():
        tile_h5.unlink()  # the float32 tile is derived data (3 GB); the crop it came from is kept
    result["seconds"] = round(time.time() - t0)
    (out / "candidates.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/processed/scan")
    parser.add_argument("--reuse", action="store_true", help="recompute candidates from saved pages/ink maps only")
    parser.add_argument("--wedge-degrees", type=float, default=40.0, help="angular cut of M2 pieces; 80 gives larger pages with the plane builder at slightly more grid shear (docs/WRAP_SPLITTING.md)")
    parser.add_argument("--split-wraps", action="store_true", help="cut touching wraps apart before flattening (docs/WRAP_SPLITTING.md); changes which pages exist")
    args = parser.parse_args()
    out_root = Path(args.out)
    shape = (768, 1024, 1024)
    tiles = {f"tile_y{y}_x{x}": (128, y, x) for y in (0, 1024) for x in (0, 1024)}
    summary = []
    for name, origin in tiles.items():
        if not args.reuse and (out_root / name / "candidates.json").exists():
            print(f"[{name}] already done, skipping", flush=True)
            summary.append(json.loads((out_root / name / "candidates.json").read_text()))
            continue
        summary.append(scan_tile(name, origin, shape, out_root, reuse=args.reuse, split_wraps=args.split_wraps, wedge_degrees=args.wedge_degrees))
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    for r in summary:
        print(r["tile"], {o: v["n_candidates"] for o, v in r["orientations"].items()}, f"{r['seconds']}s" if "seconds" in r else "", flush=True)


if __name__ == "__main__":
    main()
