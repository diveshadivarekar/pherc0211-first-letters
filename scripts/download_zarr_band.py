"""Mirror one z-band of a zarr v2 array from the Vesuvius open-data bucket, resumably.

Chunk keys use the '/' dimension separator (``<level>/<cz>/<cy>/<cx>``), so a z-band
is the set of directories ``<level>/<cz>`` whose chunk range overlaps ``[z0, z1)``.
Metadata files (``.zarray``/``.zattrs``/``.zgroup``/``metadata.json``) are always fetched.
Files already on disk with the right size are skipped.

Example:
    python scripts/download_zarr_band.py \
        --src PHerc0211/representations/predictions/surfaces/<name>.zarr --level 0 \
        --z0 7572 --z1 9328 --out data/spiral/pherc0211_src
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import s3fs

BUCKET = "vesuvius-challenge-open-data/"


def fetch(fs: s3fs.S3FileSystem, key: str, size: int, out_root: Path, strip: str) -> int:
    rel = key[len(BUCKET):]
    if strip and rel.startswith(strip):
        rel = rel[len(strip):].lstrip("/")
    dst = out_root / rel
    if dst.exists() and dst.stat().st_size == size:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(5):
        try:
            fs.get_file(key, str(tmp))
            tmp.replace(dst)
            return size
        except Exception as exc:  # network hiccup: retry with backoff
            time.sleep(2 ** attempt)
            last = exc
    raise RuntimeError(f"failed {key}: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="zarr path inside the bucket (no bucket name)")
    ap.add_argument("--level", required=True, help="pyramid level directory, e.g. 0 or 2")
    ap.add_argument("--z0", type=int, required=True, help="first z voxel IN THIS LEVEL'S coordinates")
    ap.add_argument("--z1", type=int, required=True, help="end z voxel (exclusive) in this level's coordinates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strip", default="", help="path prefix to drop when writing under --out")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    fs = s3fs.S3FileSystem(anon=True)
    base = BUCKET + args.src.rstrip("/")
    out_root = Path(args.out)
    zarray = json.loads(fs.cat(f"{base}/{args.level}/.zarray"))
    cz = zarray["chunks"][0]
    lo, hi = args.z0 // cz, (args.z1 - 1) // cz
    print(f"{args.src} level {args.level}: shape {zarray['shape']} chunks {zarray['chunks']} "
          f"-> chunk z-indices {lo}..{hi}", flush=True)

    todo: list[tuple[str, int]] = []
    for name in (".zattrs", ".zgroup", "metadata.json", f"{args.level}/.zarray", f"{args.level}/.zattrs"):
        try:
            todo.append((f"{base}/{name}", fs.info(f"{base}/{name}")["size"]))
        except FileNotFoundError:
            pass
    for z in range(lo, hi + 1):
        try:
            for key, info in fs.find(f"{base}/{args.level}/{z}", detail=True).items():
                todo.append((key, info["size"]))
        except FileNotFoundError:
            pass  # absent directory = all-fill chunks
    total = sum(s for _, s in todo)
    print(f"{len(todo)} objects, {total / 1e9:.2f} GB to consider", flush=True)

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, fs, k, s, out_root, args.strip) for k, s in todo]
        for i, f in enumerate(futs, 1):
            done += f.result()
            if i % 500 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} objects, {done / 1e9:.2f} GB new, {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
