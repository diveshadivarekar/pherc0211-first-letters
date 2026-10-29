"""Bring a real (non-synthetic) volumetric scan into the M1 pipeline format.

The project's benchmark dataset for actual ancient-scroll CT (EduceLab-
Scrolls / the Vesuvius Challenge) is 10-15 TB per scan and access-gated —
not something this offline PoC can fetch. To still exercise the pipeline on
genuinely real (not procedurally-generated) volumetric data, this module
wraps a small, freely-downloadable real 3D scan bundled with scikit-image
(``cells3d``: a real confocal microscopy z-stack of cultured cells, ~7 MB,
fetched once and cached locally) into the exact HDF5 ``volume`` format
:mod:`src.m1_ingestion.run` and the M2/M3 CLIs already expect.

Being upfront about what this does and doesn't demonstrate: it's real,
noisy, non-synthetic 3D image data, which is what matters for exercising M1
(ingestion + artifact correction) and M2 (segmentation) — both are generic
volumetric-imaging algorithms with no scroll-specific assumptions. It is
**not** a scroll, and has no ground truth (no ``layer_id``/``ink_mask``), so:

- M3's flattening will run without crashing (its cylindrical-unwrap fit
  works on any blob shape) but the result isn't meaningful — cell nuclei
  aren't developable cylindrical sheets, so there's nothing for ARAP to
  correctly flatten. Verified: it also surfaces the NaN-propagation edge
  case documented in flatten.py's arap_relax — some real cell segments
  don't span the full depth range a phantom layer always does, and their
  distortion metric comes back NaN. Non-fatal, but a real example of why
  that limitation is documented rather than assumed away.
- M4 needs ``ink_mask`` ground truth to *train* its CNN (this PoC's
  stand-in for a pretrained ink-detection model). Without it, ``run()``
  automatically falls back to the zero-dependency density-threshold
  baseline — verified this doesn't crash, it just isn't a trained model,
  since there's nothing here to train it against.
- M5 doesn't touch the input volume at all (it trains on its own synthetic
  text corpus), so it runs regardless.

Usage:
    python -m src.m1_ingestion.real_scan --out data/real_scans/cells3d.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.m1_ingestion.generate_phantom import RAW_DATASET
from src.m1_ingestion.io_utils import VolumeStore


def fetch_real_scan(channel: int = 0) -> np.ndarray:
    """Load scikit-image's ``cells3d`` sample: a real confocal microscopy
    z-stack, shape (60, 2, 256, 256) — axis 1 is (membrane, nuclei) channel.
    Downloaded and cached by ``pooch`` on first call.

    Returns a single channel as a float32 volume normalised to [0, 1],
    matching the value range M1's correction routines expect.
    """
    import skimage.data

    volume = skimage.data.cells3d()[:, channel, :, :]
    volume = volume.astype(np.float32)
    volume /= max(volume.max(), 1e-6)
    return volume


def save_real_scan(path: str | Path, volume: np.ndarray, chunk_depth: int = 16) -> Path:
    """Write ``volume`` as a plain M1-compatible HDF5 file (just the
    ``volume`` dataset — no ground truth, since none exists for real data)."""
    path = Path(path)
    chunk_depth = min(chunk_depth, volume.shape[0])
    store = VolumeStore.create(path, shape=volume.shape, dtype=np.float32, chunk_depth=chunk_depth)
    store.write_chunk(slice(0, volume.shape[0]), volume)
    store.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output HDF5 path")
    parser.add_argument("--channel", type=int, default=0, choices=[0, 1], help="0=membrane, 1=nuclei")
    args = parser.parse_args()

    volume = fetch_real_scan(channel=args.channel)
    out_path = save_real_scan(args.out, volume)
    print(f"Real scan written to {out_path}  shape={volume.shape}  dtype={volume.dtype}")
    print(f"Dataset name: '{RAW_DATASET}' — feed this file directly to: python -m src.m1_ingestion.run --input {out_path} --out <dir>")


if __name__ == "__main__":
    main()
