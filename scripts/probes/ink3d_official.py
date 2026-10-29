"""Run the official Paris 4 3D ink model (scrollprize/ink_3d_dino_guided) on a local native CT crop and score it against the Paris 4 ink labels.

Purpose: our own Paris 4 evaluation gave chance-level AUCs. This is the known-working reference model on the same 2.4 um data and the same labels. Caveat: its
training teacher used Paris 4 labels, so a good score is an in-sample sanity check of our data/label/coordinate handling, not a generalisation claim.

Step 1 (GPU, ~10-20 min): sliding-window 256^3 inference over the crop -> float16 probability volume (.npy).
Step 2 (CPU): sample the volume at every supervised label point, report AUC, a label-shift null (as ink_spatial_null.py) and a depth sweep along z.

Usage (inkenv python, from the repo root):
    D:/divesh-github/inkenv/Scripts/python.exe -u scripts/probes/ink3d_official.py infer data/real_scans/paris4/native_crop.h5 D:/divesh-github/dino3d-weights/ckpt_78k_fullsup.pth data/processed/ink3d_official/w00_prob.npy
    D:/divesh-github/inkenv/Scripts/python.exe -u scripts/probes/ink3d_official.py score data/processed/ink3d_official/w00_prob.npy data/real_scans/paris4/w00 w00_20231016151002 --out results/ink3d_official/w00.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

TILE_ORIGIN = np.array([62080, 17280, 18816], float)
TILE_SHAPE = np.array([1536, 1408, 768], float)


def infer(args: argparse.Namespace) -> None:
    import h5py
    import torch
    from vesuvius.ink_detection.data.normalization import normalize_image
    from vesuvius.ink_detection.models.checkpoint import config_from_checkpoint, load_checkpoint, load_model_state, select_inference_weights
    from vesuvius.ink_detection.models.model import make_model
    from vesuvius.ink_detection.inference.inference_runtime import TargetModel

    ck = load_checkpoint(args.checkpoint)
    cfg = config_from_checkpoint(ck, source=args.checkpoint)
    name, state = select_inference_weights(ck, source=args.checkpoint)
    print("weights:", name, "step", ck.get("step"), flush=True)
    net = make_model(cfg)
    load_model_state(net, state)
    model = TargetModel(net).cuda().eval()

    with h5py.File(args.volume, "r") as f:
        vol = f["volume"][...]
    Z, Y, X = vol.shape
    P = args.patch
    stride = max(1, int(P * (1 - args.overlap)))

    def starts(n: int) -> list[int]:
        s = list(range(0, max(n - P, 0) + 1, stride))
        if s[-1] + P < n:
            s.append(n - P)
        return s

    # separable cosine window so patch borders (little context) count less
    w1 = np.hanning(P + 2)[1:-1].astype(np.float32) ** 0.5
    win = torch.from_numpy(w1[:, None, None] * w1[None, :, None] * w1[None, None, :]).cuda()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    acc = np.lib.format.open_memmap(str(args.out) + ".acc.npy", mode="w+", dtype=np.float32, shape=vol.shape)
    wsum = np.lib.format.open_memmap(str(args.out) + ".w.npy", mode="w+", dtype=np.float32, shape=vol.shape)
    zs, ys, xs = starts(Z), starts(Y), starts(X)
    n, t0 = len(zs) * len(ys) * len(xs), time.time()
    done = 0
    for z in zs:
        for y in ys:
            for x in xs:
                patch = vol[z : z + P, y : y + P, x : x + P]
                pn = normalize_image(patch, cfg.data.normalization)
                inp = torch.from_numpy(pn)[None, None].cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(inp)
                prob = torch.sigmoid(logit.float())[0, 0] * win
                acc[z : z + P, y : y + P, x : x + P] += prob.cpu().numpy()
                wsum[z : z + P, y : y + P, x : x + P] += win.cpu().numpy()
                done += 1
                if done % 10 == 0 or done == n:
                    print(f"{done}/{n} patches {time.time() - t0:.0f}s", flush=True)
    out = np.lib.format.open_memmap(str(args.out), mode="w+", dtype=np.float16, shape=vol.shape)
    for z in range(0, Z, 64):
        out[z : z + 64] = (acc[z : z + 64] / np.maximum(wsum[z : z + 64], 1e-6)).astype(np.float16)
    out.flush()
    del acc, wsum
    for suf in (".acc.npy", ".w.npy"):
        Path(str(args.out) + suf).unlink()
    print("saved", args.out, flush=True)


def auc(s: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 1000 or y.all() or not y.any():
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sample(prob: np.ndarray, xyz: np.ndarray, dz: int = 0, half: int = 0) -> np.ndarray:
    """Max of the volume in a (2*half+1)^3 neighbourhood around each point, shifted dz voxels in z."""
    p = np.rint(xyz).astype(int)
    p[:, 0] += dz
    p = np.clip(p, 0, np.array(prob.shape) - 1)
    if half == 0:
        return prob[p[:, 0], p[:, 1], p[:, 2]].astype(np.float32)
    out = np.full(len(p), -1.0, np.float32)
    for a in range(-half, half + 1):
        for b in range(-half, half + 1):
            for c in range(-half, half + 1):
                q = np.clip(p + [a, b, c], 0, np.array(prob.shape) - 1)
                out = np.maximum(out, prob[q[:, 0], q[:, 1], q[:, 2]].astype(np.float32))
    return out


def edge_band(s: np.ndarray, lp: dict, shifts: int, bands=(16, 32, 64), hp_sigma=48.0, min_points=2000) -> dict:
    """Same statistic and shift null as scripts/probes/ink_edge_band.py, on a per-label-point score vector."""
    from probes.ink_edge_band import band_image, highpass

    ink_img, sup_img, rc = lp["ink_img"], lp["sup_img"], lp["rowcol"]
    h, w = ink_img.shape
    s = s.astype(np.float64)
    s_hp = highpass(rc, s, (h, w), hp_sigma)
    bimgs = {b: band_image(ink_img, sup_img, b) for b in bands}

    def stats(r, c):
        out = {}
        for b, bi in bimgs.items():
            m = sup_img[r, c] & bi[r, c]
            y = ink_img[r, c]
            out[b] = (auc(s[m], y[m]), auc(s_hp[m], y[m]), int(m.sum())) if m.sum() >= min_points else (np.nan, np.nan, int(m.sum()))
        return out

    obs = stats(rc[:, 0], rc[:, 1])
    rng = np.random.default_rng(0)
    null = {b: ([], []) for b in bands}
    tries = 0
    while min(len(v[0]) for v in null.values()) < shifts and tries < shifts * 10:
        tries += 1
        while True:
            dr, dc = int(rng.integers(-h, h)), int(rng.integers(-w, w))
            if max(abs(dr), abs(dc)) >= 400:
                break
        for b, (x, xh, _) in stats((rc[:, 0] + dr) % h, (rc[:, 1] + dc) % w).items():
            if np.isfinite(x) and len(null[b][0]) < shifts:
                null[b][0].append(x)
                null[b][1].append(xh)
    res = {}
    for b in bands:
        row = {"points": obs[b][2]}
        for name, k in (("band", 0), ("band_hp", 1)):
            nl, o = np.array(null[b][k]), obs[b][k]
            if len(nl) < 10 or not np.isfinite(o):
                row[name] = {"observed": float(o), "null_n": int(len(nl))}
                continue
            row[name] = {"observed": float(o), "null_n": int(len(nl)), "null_p05": float(np.percentile(nl, 5)), "null_p95": float(np.percentile(nl, 95)),
                         "p_two_sided": float((np.sum(np.abs(nl - 0.5) >= abs(o - 0.5)) + 1) / (len(nl) + 1))}
        res[str(b)] = row
    return res


def score(args: argparse.Namespace) -> None:
    from eval_ink_labels import label_points

    prob = np.load(args.prob, mmap_mode="r")
    origin, shape = TILE_ORIGIN, TILE_SHAPE
    if args.tile_json:
        tj = json.loads(args.tile_json.read_text())
        origin, shape = np.array(tj["origin_zyx"], float), np.array(tj["shape_zyx"], float)
    lp = label_points(args.labels_dir, args.prefix, origin, shape)
    xyz, ink = lp["xyz_local"], lp["ink"]
    print("label points", len(ink), "ink share", float(ink.mean()), flush=True)
    res = {"n": int(len(ink)), "ink_share": float(ink.mean())}

    # depth sweep: the labels were drawn on an older scan, so the surface can sit a few voxels off; the 3D model needs no surface
    sweep = {}
    for half, dzs in ((0, (0,)), (4, (-16, -8, 0, 8, 16))):
        for dz in dzs:
            sweep[f"half{half}_dz{dz}"] = auc(sample(prob, xyz, dz, half), ink)
    res["sweep_auc"] = sweep
    print(json.dumps(sweep, indent=1), flush=True)

    # best-looking setting vs a label-shift null (shifts the labels inside the canvas, re-scores the same points)
    key = "half4_dz0"  # fixed up front from the w00 run, not re-picked per segment
    half, dz = int(key.split("_")[0][4:]), int(key.split("dz")[1])
    s = sample(prob, xyz, dz, half)
    res["chosen"], res["auc_chosen"] = key, auc(s, ink)
    r0, c0, r1, c1 = lp["window"]
    sup, inkimg = lp["sup_img"], lp["ink_img"]
    H, W = inkimg.shape
    rc = lp["rowcol"]
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(args.shifts):
        ang = rng.uniform(0, 2 * np.pi)
        mag = rng.uniform(args.min_shift, 2.5 * args.min_shift)
        dr, dc = int(mag * np.sin(ang)), int(mag * np.cos(ang))
        rr, cc = (rc[:, 0] + dr) % H, (rc[:, 1] + dc) % W
        ok = sup[rr, cc]
        if ok.sum() < 20000:
            continue
        nulls.append(auc(s[ok], inkimg[rr, cc][ok]))
    nulls = np.array([v for v in nulls if np.isfinite(v)])
    res["null_n"] = int(len(nulls))
    if len(nulls):
        res["null_5_95"] = [float(np.percentile(nulls, 5)), float(np.percentile(nulls, 95))]
        res["p_two_sided"] = float((np.abs(nulls - 0.5) >= abs(res["auc_chosen"] - 0.5)).mean())
    res["edge_band"] = edge_band(s, lp, args.shifts)
    print(json.dumps({k: v for k, v in res.items() if k != "sweep_auc"}, indent=1), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("infer")
    a.add_argument("volume", type=Path)
    a.add_argument("checkpoint", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--patch", type=int, default=256)
    a.add_argument("--overlap", type=float, default=0.5)
    a.set_defaults(fn=infer)
    b = sub.add_parser("score")
    b.add_argument("prob", type=Path)
    b.add_argument("labels_dir", type=Path)
    b.add_argument("prefix")
    b.add_argument("--tile-json", type=Path, default=None, help="origin/shape of the crop (default: the w00 tile)")
    b.add_argument("--shifts", type=int, default=200)
    b.add_argument("--min-shift", type=int, default=400)
    b.add_argument("--out", type=Path, default=None)
    b.set_defaults(fn=score)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
