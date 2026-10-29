"""Pick the evaluation tile for a labelled segment: the densest, well-supervised block of ink labels and the CT crop around it.

Reads the segment's ink labels, supervision mask and tifxyz geometry (never the surface volume). Blocks of ``--block`` canvas pixels are ranked
by the number of supervised ink pixels they hold (only blocks that are mostly supervised count); the first block whose 3D box, grown by
``--margin`` voxels and rounded out to the zarr's 128-voxel chunks, fits inside the volume and under ``--max-gib`` is chosen. Prints the
tile as ``--z0 --y0 --x0 --depth --ysize --xsize`` for ``download_native_crop.py`` (the surface-prediction crop is the same box in units of its scale).

Usage:
    python scripts/pick_label_tile.py --labels-dir data/real_scans/paris4/w02 --prefix w02_20231031143852 --out data/real_scans/paris4/w02/tile.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

Image.MAX_IMAGE_PIXELS = None
CHUNK = 128
VOLUME_SHAPE = (75784, 32693, 32693)  # PHercParis4 2.4 um, level 0 (z, y, x)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--block", type=int, default=1024, help="block edge in canvas pixels")
    p.add_argument("--margin", type=float, default=64.0)
    p.add_argument("--max-gib", type=float, default=2.5)
    p.add_argument("--min-supervised", type=float, default=0.8)
    p.add_argument("--volume-shape", type=int, nargs=3, default=VOLUME_SHAPE)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    meta = json.loads((a.labels_dir / "meta.json").read_text())
    sx, sy = meta["scale"]
    ink = Image.open(a.labels_dir / f"{a.prefix}_inklabels.tif")
    sup = Image.open(a.labels_dir / f"{a.prefix}_supervision_mask.tif")
    print(f"canvas {ink.size[0]} x {ink.size[1]} (w x h), grid scale {meta['scale']}")
    inksup = np.asarray(ImageChops.darker(ink, sup).reduce(a.block)).astype(np.float64) / 255.0 * a.block**2  # supervised ink pixels per block
    supfrac = np.asarray(sup.reduce(a.block)).astype(np.float64) / 255.0
    total_ink = float(np.asarray(ink.reduce(a.block)).astype(np.float64).sum() / 255.0 * a.block**2)
    print(f"ink pixels ~{total_ink:,.0f}; supervised fraction of canvas {supfrac.mean():.3f}")
    g = {k: np.asarray(Image.open(a.labels_dir / f"{k}.tif")).astype(np.float64) for k in "xyz"}
    order = np.dstack(np.unravel_index(np.argsort(-(inksup * (supfrac >= a.min_supervised)), axis=None), inksup.shape))[0]
    vol = np.array(a.volume_shape, float)
    for bi, bj in order[:200]:
        if inksup[bi, bj] <= 0 or supfrac[bi, bj] < a.min_supervised:
            break
        r0, r1 = int(bi * a.block * sy), int((bi + 1) * a.block * sy) + 1
        c0, c1 = int(bj * a.block * sx), int((bj + 1) * a.block * sx) + 1
        px, py, pz = (g[k][r0:r1, c0:c1] for k in "xyz")
        ok = (pz > 0) & np.isfinite(pz)
        if ok.sum() < 100:
            continue
        lo = np.array([pz[ok].min(), py[ok].min(), px[ok].min()]) - a.margin
        hi = np.array([pz[ok].max(), py[ok].max(), px[ok].max()]) + a.margin
        lo = np.floor(lo / CHUNK) * CHUNK
        hi = np.ceil(hi / CHUNK) * CHUNK
        shape = hi - lo
        gib = shape.prod() / 2**30
        if (lo < 0).any() or (hi > vol).any() or gib > a.max_gib:
            print(f"  block ({bi},{bj}) skipped: {shape.astype(int).tolist()} = {gib:.1f} GiB")
            continue
        tile = {
            "block": [int(bi), int(bj)], "block_pixels": a.block, "supervised_ink_pixels": float(inksup[bi, bj]), "supervised_fraction": float(supfrac[bi, bj]),
            "origin_zyx": [int(v) for v in lo], "shape_zyx": [int(v) for v in shape], "gib_uint8": float(gib),
        }
        print(f"tile: block ({bi},{bj}), {inksup[bi, bj]:,.0f} supervised ink px, {supfrac[bi, bj]:.2f} supervised; origin z/y/x {tile['origin_zyx']}, shape {tile['shape_zyx']} = {gib:.2f} GiB")
        print(f"  --z0 {tile['origin_zyx'][0]} --y0 {tile['origin_zyx'][1]} --x0 {tile['origin_zyx'][2]} --depth {tile['shape_zyx'][0]} --ysize {tile['shape_zyx'][1]} --xsize {tile['shape_zyx'][2]}")
        if a.out:
            a.out.write_text(json.dumps(tile, indent=1))
        return
    raise SystemExit("no block fits the size limit; raise --max-gib or lower --block")


if __name__ == "__main__":
    main()
