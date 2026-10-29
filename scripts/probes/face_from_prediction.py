"""Extract flat sheet faces from a surface-prediction crop without a scroll axis, and render them from the native CT.

For each large, flat connected component of the prediction (coarse voxels, ``scale`` native voxels each) the component's
principal plane gives a normal; its voxels are binned into a height field over that plane, snapped to the papyrus ridge in
the native CT (+-``--snap`` voxels), and the CT is sampled at several offsets along the normal. Output is one PNG per
component (several depth offsets side by side, 1 mm scale bar) plus a JSON of component statistics.

This is a *look* at one face, not a flattened mesh: it assumes the surface is single-valued over its principal plane, and
that breaks for folds. A registration number is reported: the mean CT along the *unsnapped* surface vs offset.

Usage:
    python scripts/probes/face_from_prediction.py --crop data/real_scans/pherc0846a/native_crop_W0.h5 \
        --pred data/real_scans/pherc0846a/surface_pred_crop_W0.h5 --out data/processed/pherc0846a_faces --top 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


def stretch(a: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    v = a[valid] if valid is not None and valid.any() else a.ravel()
    lo, hi = np.percentile(v, [1, 99.5])
    t = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    if valid is not None:
        t = t * valid
    return (t * 255).astype(np.uint8)


def components(mask: np.ndarray, min_voxels: int) -> list[dict]:
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        coords = np.argwhere(lab[sl] == i) + np.array([s.start for s in sl])
        if len(coords) < min_voxels:
            continue
        mu = coords.mean(0)
        w, v = np.linalg.eigh(np.cov((coords - mu).T))      # ascending
        out.append({"coords": coords, "mu": mu, "n": v[:, 0], "w": v[:, 1], "u": v[:, 2],
                    "flat": float(np.sqrt(max(w[0], 0) / max(w[1], 1e-9))),
                    "extent": float(np.sqrt(w[2] * w[1])), "voxels": int(len(coords))})
    return sorted(out, key=lambda c: -c["extent"] * (c["flat"] < 0.3))


def height_field(c: dict, scale: int, sigma: float = 1.0):
    """Native-resolution height field over the component's plane. Returns (h, valid, s0, t0)."""
    d = c["coords"] - c["mu"]
    s, t, h = d @ c["u"], d @ c["w"], d @ c["n"]
    s0, t0 = np.floor(s.min()), np.floor(t.min())
    si, ti = (s - s0).astype(int), (t - t0).astype(int)
    S, T = si.max() + 1, ti.max() + 1
    acc, cnt = np.zeros((S, T)), np.zeros((S, T))
    np.add.at(acc, (si, ti), h)
    np.add.at(cnt, (si, ti), 1)
    occ = cnt > 0
    hm = np.where(occ, acc / np.maximum(cnt, 1), np.nan)
    idx = ndi.distance_transform_edt(~occ, return_distances=False, return_indices=True)
    filled = hm[idx[0], idx[1]]
    filled = ndi.gaussian_filter(filled, sigma)
    valid = ndi.binary_closing(occ, iterations=2)
    hf = ndi.zoom(filled, scale, order=1)
    vf = ndi.zoom(valid.astype(np.float32), scale, order=1) > 0.5
    return hf * scale, vf, s0 * scale, t0 * scale


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--threshold", type=int, default=127)
    p.add_argument("--min-voxels", type=int, default=400, help="coarse voxels")
    p.add_argument("--snap", type=int, default=10, help="native voxels searched each side to snap onto the ridge")
    p.add_argument("--header", default=None, help="text at the top of every PNG (scroll, volume ID, voxel size); default is the crop name")
    p.add_argument("--offsets", type=int, nargs="+", default=[-6, -3, 0, 3, 6])
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.pred, "r") as g:
        scale = int(g["volume"].attrs["scale_to_native"])
        mask = np.asarray(g["volume"]) > a.threshold
    with h5py.File(a.crop, "r") as f:
        ct = np.asarray(f["volume"], dtype=np.float32)
        um = float(f["volume"].attrs.get("voxel_size_um", f.attrs.get("voxel_size_um", 2.403)))
    ct_s = ndi.gaussian_filter(ct, 1.0)
    comps = components(mask, a.min_voxels)
    print(f"{len(comps)} components >= {a.min_voxels} coarse voxels; scale {scale}; voxel {um} um", flush=True)
    summary = []
    for rank, c in enumerate(comps[: a.top]):
        h, valid, s0, t0 = height_field(c, scale)
        S, T = h.shape
        ss, tt = np.meshgrid(np.arange(S) + s0, np.arange(T) + t0, indexing="ij")
        mu_nat = c["mu"] * scale + (scale - 1) / 2
        base = mu_nat[:, None, None] + ss * c["u"][:, None, None] + tt * c["w"][:, None, None]

        def sample(vol: np.ndarray, off: np.ndarray) -> np.ndarray:
            pts = base + (h + off)[None] * c["n"][:, None, None]
            return ndi.map_coordinates(vol, pts, order=1, mode="constant", cval=0.0)

        # registration of the raw (unsnapped) surface
        d_grid = np.arange(-a.snap, a.snap + 1)
        prof = np.stack([sample(ct_s, np.full_like(h, d))[valid].mean() for d in d_grid])
        # snap: per-pixel argmax of a locally averaged profile
        stack = np.stack([ndi.uniform_filter(sample(ct_s, np.full_like(h, d)), 15) for d in d_grid])
        delta = ndi.median_filter(d_grid[np.argmax(stack, 0)].astype(np.float32), 21)
        hs = h + delta
        tiles = []
        for off in a.offsets:
            img = sample(ct, hs + off)
            tiles.append(stretch(img, valid & (img > 0)))
        # orient so the long side is horizontal
        if tiles[0].shape[0] > tiles[0].shape[1]:
            tiles = [t.T for t in tiles]
        H = tiles[0].shape[0]
        row = np.concatenate([np.pad(t, ((0, 0), (0, 6)), constant_values=255) for t in tiles], 1)
        im = Image.fromarray(row).convert("RGB")
        bar = int(round(1000.0 / um))
        canvas = Image.new("RGB", (max(im.width, bar + 140), im.height + 56), "white")
        canvas.paste(im, (0, 24))
        d = ImageDraw.Draw(canvas)
        d.text((4, 6), f"{a.header or Path(a.crop).stem} | component {rank} | offsets {a.offsets} vox along normal", fill="black")
        y = 24 + im.height + 14
        d.rectangle([6, y, 6 + bar, y + 5], fill="black")
        d.text((14 + bar, y - 3), "1 mm", fill="black")
        name = out / f"face_{rank}.png"
        canvas.save(name)
        area_mm2 = float(valid.sum()) * (um / 1000) ** 2
        rec = {"rank": rank, "file": name.name, "coarse_voxels": c["voxels"], "flatness": round(c["flat"], 3),
               "area_mm2": round(area_mm2, 2), "normal_zyx": [round(float(v), 3) for v in c["n"]],
               "mean_ct_vs_offset_unsnapped": {int(o): round(float(v), 1) for o, v in zip(d_grid, prof)},
               "peak_offset_unsnapped": int(d_grid[int(np.argmax(prof))]),
               "snap_delta_mean_std_p5_p95": [round(float(x), 2) for x in (delta[valid].mean(), delta[valid].std(),
                                                                          *np.percentile(delta[valid], [5, 95]))]}
        summary.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "mean_ct_vs_offset_unsnapped"}), flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "faces.json").write_text(json.dumps({"header": a.header, "crop": a.crop, "pred": a.pred, "git_commit": commit,
                                                "voxel_um": um, "faces": summary}, indent=1))


if __name__ == "__main__":
    main()
