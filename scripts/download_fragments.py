"""Download the Vesuvius Challenge ink-detection fragments (surface volumes + ink labels).

Source: https://dl.ash2txt.org/fragments/ (public, no login). Each fragment folder holds
``surface_volume/00.tif .. 64.tif`` (65 slices sampled across the papyrus surface), a binary
``inklabels.png`` (ground-truth ink) and ``mask.png`` (which pixels are papyrus).

Files are fetched in parallel, resumed if partly downloaded, and skipped if already complete
(size checked against the server), so the script is safe to re-run after an interruption.

Usage:
    python scripts/download_fragments.py --out data/real_scans/fragments --fragments 1 2 3
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://dl.ash2txt.org/fragments"
FRAGMENT_DIRS = {1: "Frag1/PHercParis2Fr47.volpkg", 2: "Frag2/PHercParis2Fr143.volpkg", 3: "Frag3/PHercParis1Fr34.volpkg"}
SURFACE = "working/54keV_exposed_surface"
LABEL_FILES = ("inklabels.png", "mask.png")
CHUNK = 1 << 20


def _head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def _list_tifs(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="([^"?/][^"]*\.tif)"', html)))


def fetch(url: str, dest: Path, retries: int = 5) -> int:
    """Download ``url`` to ``dest`` (resuming a partial ``.part`` file); returns bytes of the final file."""
    expected = _head_size(url)
    if dest.exists() and dest.stat().st_size == expected:
        return expected
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > expected:
                part.unlink()
                have = 0
            if have < expected:
                req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
                with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                    if have and r.status != 206:  # server ignored the range request: start over
                        f.truncate(0)
                    while chunk := r.read(CHUNK):
                        f.write(chunk)
            if part.stat().st_size == expected:
                part.replace(dest)
                return expected
        except OSError as exc:
            print(f"  retry {attempt + 1}/{retries} for {dest.name}: {exc}", flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/real_scans/fragments")
    parser.add_argument("--fragments", type=int, nargs="+", default=[1, 2, 3], choices=sorted(FRAGMENT_DIRS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs: list[tuple[str, Path]] = []
    for n in args.fragments:
        root = f"{BASE}/{FRAGMENT_DIRS[n]}/{SURFACE}"
        out = Path(args.out) / f"frag{n}"
        for name in LABEL_FILES:
            jobs.append((f"{root}/{name}", out / name))
        for name in _list_tifs(f"{root}/surface_volume/"):
            jobs.append((f"{root}/surface_volume/{name}", out / "surface_volume" / name))

    with ThreadPoolExecutor(args.workers) as pool:
        total = sum(pool.map(lambda job: _head_size(job[0]), jobs))
    print(f"{len(jobs)} files, {total / 2**30:.1f} GiB -> {args.out}", flush=True)
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, u, d): d for u, d in jobs}
        for fut in as_completed(futures):
            done += fut.result()
            rate = done / max(time.time() - t0, 1e-6) / 2**20
            print(f"  {done / 2**30:.2f}/{total / 2**30:.1f} GiB  ({rate:.1f} MiB/s)  {futures[fut].name}", flush=True)
    print("Download complete.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
