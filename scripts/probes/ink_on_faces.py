"""Run the fragment-trained ink U-Net (M4) on faces cut from a prediction crop, with shifted-surface controls.

Builds a ``*_flattened.h5`` in the layout M4 expects (one ``layer_*`` per surface: ``page`` + ``coord_map``) from the flat
components of a surface-prediction crop (same geometry as face_from_prediction.py), adds copies of each surface shifted by
``--shifts`` voxels along its normal as controls, runs ``run_fragment_model`` in both depth orientations, and prints the
per-surface ink-probability statistics. There is NO ground truth here: this checks whether the output differs between the
surface and its controls and whether it forms stroke-like structure, nothing more. The model is trained on 54 keV / 3.24 um
fragments; on any other scan its output is a candidate generator only.

Usage:
    python scripts/probes/ink_on_faces.py --crop data/real_scans/pherc0846a/native_crop_W0.h5 \
        --pred data/real_scans/pherc0846a/surface_pred_crop_W0.h5 --model data/processed/models/ink_unet_20k.pt \
        --faces 0 1 3 --out data/processed/pherc0846a_ink
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_from_prediction import components, height_field  # noqa: E402
from src.m4_ink_detection.run import run_fragment_model  # noqa: E402


def coord_map_for(c: dict, scale: int, shift: float = 0.0):
    h, valid, s0, t0 = height_field(c, scale)
    S, T = h.shape
    ss, tt = np.meshgrid(np.arange(S) + s0, np.arange(T) + t0, indexing="ij")
    mu_nat = c["mu"] * scale + (scale - 1) / 2
    base = mu_nat[:, None, None] + ss * c["u"][:, None, None] + tt * c["w"][:, None, None]
    pts = base + (h + shift)[None] * c["n"][:, None, None]
    cm = np.moveaxis(pts, 0, -1).astype(np.float32)          # (S, T, 3) in crop (z, y, x)
    cm[~valid] = np.nan
    return cm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--faces", type=int, nargs="+", default=[0])
    p.add_argument("--shifts", type=float, nargs="+", default=[30.0, -30.0])
    p.add_argument("--voxel-um", type=float, default=2.403)
    p.add_argument("--threshold", type=int, default=127)
    p.add_argument("--min-voxels", type=int, default=400)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.pred, "r") as g:
        scale = int(g["volume"].attrs["scale_to_native"])
        mask = np.asarray(g["volume"]) > a.threshold
    comps = components(mask, a.min_voxels)
    flat = out / "faces_flattened.h5"
    names: dict[str, str] = {}
    with h5py.File(flat, "w") as f:
        k = 0
        for r in a.faces:
            for shift in [0.0, *a.shifts]:
                k += 1
                cm = coord_map_for(comps[r], scale, shift)
                grp = f.create_group(f"layer_{k}")
                grp.create_dataset("coord_map", data=cm, compression="lzf")
                grp.create_dataset("page", data=np.zeros(cm.shape[:2], np.float32), compression="lzf")
                grp.create_dataset("uv", data=np.zeros(cm.shape, np.float32), compression="lzf")
                names[f"layer_{k}"] = f"face{r}" + ("" if shift == 0 else f"_shift{shift:+g}")
    result = {"model": a.model, "voxel_um": a.voxel_um, "validated": False, "layers": {}}
    for orient in ("normal", "flip"):
        odir = out / f"ink_{orient}"
        odir.mkdir(exist_ok=True)
        ink = run_fragment_model(flat, a.crop, odir, a.model, voxel_size_um=a.voxel_um, max_pages=64,
                                 flip_depth=orient == "flip", refine=True)
        with h5py.File(ink, "r") as fi:
            for layer in fi:
                pr = fi[layer]["ink_probability"][...]
                v = ~np.isnan(fi[layer]["coord_map"][..., 0]) & (fi[layer]["page"][...] > 0)
                vals = pr[v] if v.any() else np.array([0.0])
                result["layers"].setdefault(names[layer], {})[orient] = {
                    "valid_px": int(v.sum()), "mean": round(float(vals.mean()), 4),
                    "p95": round(float(np.percentile(vals, 95)), 4), "p99": round(float(np.percentile(vals, 99)), 4),
                    "frac_gt_0.5": round(float((vals > 0.5).mean()), 4)}
    (out / "ink_stats.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["layers"], indent=1))


if __name__ == "__main__":
    main()
