"""Turn a VC3D-rendered tifxyz page + its ``vesuvius`` ink inference TIFF into the page ``.h5`` that ``eval_ink_labels.py`` reads.

``vc_render_tifxyz`` renders a surface at ``1/scale`` pixels per grid cell, so rendered pixel (row, col) sits at tifxyz grid
(row * scale_y, col * scale_x). The tifxyz x/y/z grids are bilinearly sampled there to give a ``coord_map`` (z, y, x) per pixel.

Usage:
    python scripts/ink_from_vc3d.py --tifxyz data/processed/tifxyz/paris4_split/layer_2 --ink data/processed/paris4/ink9um/layer_2.tif \
        --out data/processed/paris4/ink9um/layer_2.h5 --name layer_2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import tifffile
from scipy.ndimage import map_coordinates


def build(tifxyz: Path, ink: Path, out: Path, name: str, mode: str = "a", offset=(0.0, 0.0, 0.0), render_scale: float = 1.0) -> None:
    scale_x, scale_y = json.loads((tifxyz / "meta.json").read_text())["scale"]
    xyz = [tifffile.imread(tifxyz / f"{k}.tif").astype(np.float64) for k in "xyz"]
    prob = tifffile.imread(ink)
    prob = prob.astype(np.float32) / (255.0 if prob.dtype == np.uint8 else 1.0)
    h, w = prob.shape
    rr, cc = np.meshgrid(np.arange(h) / render_scale * scale_y, np.arange(w) / render_scale * scale_x, indexing="ij")
    valid = (xyz[0] > 0) & (xyz[1] > 0) & (xyz[2] > 0)
    near = map_coordinates(valid.astype(np.float32), [rr, cc], order=1, mode="nearest") > 0.999
    coord = np.stack([np.where(near, map_coordinates(np.where(valid, g, 0.0), [rr, cc], order=1, mode="nearest"), np.nan) for g in (xyz[2], xyz[1], xyz[0])], -1)
    coord = coord - np.asarray(offset, dtype=np.float64)  # scroll voxels -> tile-local (z, y, x)
    with h5py.File(out, mode) as f:
        if name in f:
            del f[name]
        g = f.create_group(name)
        g.create_dataset("coord_map", data=coord.astype(np.float32), compression="gzip")
        g.create_dataset("ink_probability", data=prob, compression="gzip")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tifxyz", type=Path, required=True)
    p.add_argument("--ink", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="page")
    p.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("Z", "Y", "X"), help="tile origin in scroll voxels, subtracted to give tile-local coordinates")
    p.add_argument("--render-scale", type=float, default=1.0, help="vc_render_tifxyz --scale used (pixels per level-0 voxel)")
    a = p.parse_args()
    build(a.tifxyz, a.ink, a.out, a.name, "w", a.offset, a.render_scale)
    print(f"wrote {a.out}")
