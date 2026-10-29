"""Bring a real, cropped piece of an actual Herculaneum scroll into the M1
pipeline format.

Unlike real_scan.py's microscopy stand-in, this is the genuine article: the
Vesuvius Challenge's open-data S3 bucket hosts real synchrotron micro-CT
scans of carbonised scrolls, publicly readable with no account/agreement
(verified interactively — ``s3fs.S3FileSystem(anon=True)`` against
``s3://vesuvius-challenge-open-data/``), stored as OME-Zarr multiscale
volumes exactly matching the format ARCHITECTURE.md specifies for this
project. A full scroll at native resolution is 8+ TB; this module picks a
downsampled multiscale level and a z-range crop to bring that down to a few
GB, streamed chunk-by-chunk so the download never needs the whole crop
resident in memory at once.

Read this before expecting ink/text results: the bucket's per-scroll data
has no ground-truth ink labels anywhere (checked ``PHerc0332`` and
``PHerc0175A`` — only geometric "predictions" like fibre/surface detection,
no ink masks). That's not a gap in this loader; it's the literal unsolved
problem the Vesuvius Challenge exists to solve, and nobody has published
per-voxel ink ground truth for a full scroll. So on this data:

- M1 (ingestion/correction) and M2/M3 (segmentation/flattening) are
  genuinely meaningful here — this is real scroll geometry (an actual
  rolled/spiral cross-section, not a synthetic ring phantom or unrelated
  microscopy data), with real synchrotron noise for M1 to correct.
- M4 has nothing to train its ink CNN against here and will fall back to
  the classical density baseline, same as on real_scan.py's data — that
  baseline was never expected to find genuine carbon ink (see
  ink_detection.py's own docstring on why that needs a trained model).
- M5 doesn't touch the input volume at all.

Usage:
    python -m src.m1_ingestion.real_scroll --out data/real_scans/pherc0332_crop.h5 \
        --scroll-id PHerc0332 --level 3 --z-fraction 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.m1_ingestion.io_utils import VolumeStore

BUCKET = "vesuvius-challenge-open-data"


def _find_volume_zarr_path(scroll_id: str) -> str:
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    volumes = fs.ls(f"s3://{BUCKET}/{scroll_id}/volumes/")
    if not volumes:
        raise ValueError(f"No volumes found for scroll '{scroll_id}' in the open-data bucket")
    return volumes[0]


def inspect_levels(scroll_id: str) -> list[dict]:
    """List every OME-Zarr resolution level's shape and uncompressed size
    (this bucket's arrays are stored uncompressed, so that's also the real
    download size) — call this before picking a level/crop to download."""
    import s3fs
    import zarr

    fs = s3fs.S3FileSystem(anon=True)
    zarr_path = _find_volume_zarr_path(scroll_id)
    store = s3fs.S3Map(root=zarr_path, s3=fs, check=False)
    group = zarr.open(store, mode="r")

    levels = []
    for key in sorted(group.keys(), key=int):
        arr = group[key]
        levels.append(
            {
                "level": int(key),
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "gb": arr.nbytes / 1e9,
            }
        )
    return levels


def _fetch_slab_with_retry(arr, z0: int, z1: int, max_attempts: int = 6, base_delay: float = 2.0):
    """S3 range reads over a multi-GB streamed download will occasionally
    hit a transient connection error (verified: a real run failed with
    ``EndpointConnectionError`` ~24% of the way through, not from anything
    wrong with the data or the request). Retry with exponential backoff
    rather than letting one blip kill an hours-long download."""
    import time

    last_exc = None
    for attempt in range(max_attempts):
        try:
            return arr[z0:z1]
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any transient fetch failure should retry
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = base_delay * (2**attempt)
                print(f"  fetch of z=[{z0}:{z1}] failed ({exc}); retrying in {delay:.0f}s...", flush=True)
                time.sleep(delay)
    raise last_exc


def download_scroll_crop(
    scroll_id: str,
    out_path: str | Path,
    level: int = 3,
    z_fraction: float = 0.5,
    z_start_fraction: float = 0.25,
    stream_depth: int = 16,
    progress: bool = True,
    resume: bool = True,
) -> Path:
    """Stream a z-range crop of one OME-Zarr resolution level into an
    M1-compatible HDF5 file.

    ``z_fraction`` of the level's full depth is kept, starting at
    ``z_start_fraction`` through the stack (default: the middle half of the
    scroll, avoiding the often mostly-empty very top/bottom).

    Fetches from S3 aligned to the zarr array's own chunk depth (128 for
    this bucket's volumes) — this matters more than it looks: zarr always
    fetches whole chunks, spanning the *full* height/width, regardless of
    how thin a z-slice you ask for. Verified on a real run: requesting
    ``stream_depth``-sized (16) unaligned slabs meant each 128-deep chunk
    row got silently re-fetched 8 times over (once per 16-slice sub-read
    landing inside it) — an 8x bandwidth blow-up that made an ~8 GB crop
    crawl. Fetching one full chunk-row at a time eliminates that, at the
    cost of a larger transient buffer (~500 MB per chunk-row); ``stream_depth``
    then only controls the *write* batch size within an already-fetched
    chunk-row, keeping the float32-conversion buffer small regardless.

    Resumable by default: a multi-GB download over a real network can and
    (verified) does hit transient connection drops. If ``out_path`` already
    exists with matching scroll/level/range attrs and a ``completed_slices``
    marker, this picks up from there instead of restarting from slice 0.
    Pass ``resume=False`` to force a clean restart.
    """
    import s3fs
    import zarr

    fs = s3fs.S3FileSystem(anon=True)
    zarr_path = _find_volume_zarr_path(scroll_id)
    store = s3fs.S3Map(root=zarr_path, s3=fs, check=False)
    group = zarr.open(store, mode="r")
    arr = group[str(level)]

    full_depth, height, width = arr.shape
    z_start = int(full_depth * z_start_fraction)
    z_stop = min(full_depth, z_start + int(full_depth * z_fraction))
    depth = z_stop - z_start

    out_path = Path(out_path)
    resume_from = 0
    if resume and out_path.exists():
        existing = VolumeStore(out_path, mode="r+")
        attrs = existing.attrs
        same_request = (
            attrs.get("scroll_id") == scroll_id
            and int(attrs.get("level", -1)) == level
            and int(attrs.get("source_z_start", -1)) == z_start
            and int(attrs.get("source_z_stop", -1)) == z_stop
        )
        if same_request and existing.shape == (depth, height, width):
            resume_from = int(attrs.get("completed_slices", 0))
            volume_store = existing
            if progress and resume_from > 0:
                print(f"Resuming from slice {resume_from}/{depth} (found existing partial download)", flush=True)
        else:
            existing.close()
            volume_store = VolumeStore.create(
                out_path,
                shape=(depth, height, width),
                dtype=np.float32,
                chunk_depth=stream_depth,
                attrs={"scroll_id": scroll_id, "level": level, "source_z_start": z_start, "source_z_stop": z_stop, "completed_slices": 0},
            )
    else:
        volume_store = VolumeStore.create(
            out_path,
            shape=(depth, height, width),
            dtype=np.float32,
            chunk_depth=stream_depth,
            attrs={"scroll_id": scroll_id, "level": level, "source_z_start": z_start, "source_z_stop": z_stop, "completed_slices": 0},
        )

    if progress:
        est_gb = depth * height * width / 1e9  # source dtype is uint8
        print(
            f"Streaming {scroll_id} level {level}, z=[{z_start}:{z_stop}] ({depth} slices, ~{est_gb:.2f} GB from S3)...",
            flush=True,
        )

    chunk_z = arr.chunks[0]
    write_batch = min(stream_depth, chunk_z)
    already_done_until = z_start + resume_from
    aligned_start = (z_start // chunk_z) * chunk_z

    for chunk_start in range(aligned_start, z_stop, chunk_z):
        chunk_stop = min(chunk_start + chunk_z, full_depth)
        overlap_start = max(chunk_start, z_start, already_done_until)
        overlap_stop = min(chunk_stop, z_stop)
        if overlap_start >= overlap_stop:
            continue  # this chunk-row is entirely outside our range, or already done (resume)

        raw_chunk = _fetch_slab_with_retry(arr, chunk_start, chunk_stop)  # one S3 fetch per chunk-row

        for sub0 in range(overlap_start, overlap_stop, write_batch):
            sub1 = min(sub0 + write_batch, overlap_stop)
            local0, local1 = sub0 - chunk_start, sub1 - chunk_start
            slab = raw_chunk[local0:local1].astype(np.float32) / 255.0
            volume_store.write_chunk(slice(sub0 - z_start, sub1 - z_start), slab)
            volume_store.dataset.attrs["completed_slices"] = sub1 - z_start
            volume_store.dataset.flush()  # make progress (and the resume marker) durable, not just buffered
            if progress:
                done = sub1 - z_start
                print(f"  {done}/{depth} slices ({100 * done / depth:.1f}%)", flush=True)

        del raw_chunk  # free the ~500 MB chunk-row buffer before fetching the next one

    volume_store.close()
    if progress:
        print(f"Real scroll crop written to {out_path}", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="List a scroll's available resolution levels and sizes")
    inspect_p.add_argument("--scroll-id", required=True)

    download_p = sub.add_parser("download", help="Stream a cropped region to an M1-compatible HDF5 file")
    download_p.add_argument("--out", required=True)
    download_p.add_argument("--scroll-id", required=True)
    download_p.add_argument("--level", type=int, default=3)
    download_p.add_argument("--z-fraction", type=float, default=0.5)
    download_p.add_argument("--z-start-fraction", type=float, default=0.25)
    download_p.add_argument("--stream-depth", type=int, default=16)
    download_p.add_argument("--no-resume", action="store_true", help="Force a clean restart instead of resuming a partial download")

    args = parser.parse_args()

    if args.command == "inspect":
        for lvl in inspect_levels(args.scroll_id):
            print(f"level {lvl['level']}: shape={lvl['shape']} dtype={lvl['dtype']} size={lvl['gb']:.2f} GB")
    elif args.command == "download":
        download_scroll_crop(
            args.scroll_id,
            args.out,
            level=args.level,
            z_fraction=args.z_fraction,
            z_start_fraction=args.z_start_fraction,
            stream_depth=args.stream_depth,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
