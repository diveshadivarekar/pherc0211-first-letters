"""Export clear PNG previews of every pipeline stage — the CLI's answer to
the GUI's before/after panels, for anyone who wants to *see* the output of
scripts/run_end_to_end.py without opening the Tkinter app.

Looks for each stage's HDF5 output by the same naming convention
run_end_to_end.py produces (``<stem>_corrected.h5``, ``<stem>_segmented.h5``,
etc.) inside --out, and skips any stage whose file isn't there instead of
failing — e.g. a real (non-phantom) scan run via M1+M2 only still gets M1/M2
previews.

Usage:
    python scripts/export_previews.py --stem phantom_001 --out data/processed/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.imaging import save_side_by_side, to_display_image

PREVIEW_SIZE = (480, 480)


def _pick_demo_layer(flattened_path: Path, ink_path: Path | None) -> int:
    """Prefer a layer M4 actually found ink on (a more interesting demo
    image); otherwise fall back to layer 1."""
    if ink_path and ink_path.exists():
        with h5py.File(ink_path, "r") as f:
            best_layer, best_max = 1, -1.0
            for name in f.keys():
                prob = f[name]["ink_probability"][...]
                if prob.max() > best_max:
                    best_max = float(prob.max())
                    best_layer = int(name.split("_")[1])
            if best_max > 0.5:
                return best_layer
    with h5py.File(flattened_path, "r") as f:
        return int(sorted(f.keys(), key=lambda n: int(n.split("_")[1]))[0].split("_")[1])


def _find_raw_input(stem: str, out_dir: Path) -> Path | None:
    """Locate the original input file for the M1 before/after preview.

    Tries, in order: an explicit copy inside ``out_dir`` itself, then the
    conventional ``data/phantoms/`` and ``data/real_scans/`` locations
    relative to the current working directory (where these scripts are
    normally run from) — not relative to ``out_dir``, since ``out_dir`` is
    typically a *subdirectory* like ``data/processed/e2e_run``, not a
    sibling of ``data/phantoms``.
    """
    candidates = [
        out_dir / f"{stem}.h5",
        Path("data/phantoms") / f"{stem}.h5",
        Path("data/real_scans") / f"{stem}.h5",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def export_previews(
    stem: str, out_dir: str | Path, slice_idx: int | None = None, raw_path: str | Path | None = None
) -> list[Path]:
    out_dir = Path(out_dir)
    written: list[Path] = []

    raw_path = Path(raw_path) if raw_path else _find_raw_input(stem, out_dir)

    corrected_path = out_dir / f"{stem}_corrected.h5"
    segmented_path = out_dir / f"{stem}_segmented.h5"
    flattened_path = out_dir / f"{stem}_flattened.h5"
    ink_path = out_dir / f"{stem}_ink.h5"

    if slice_idx is None:
        # Default to the middle of the stack, not slice 0 — for a uniform
        # phantom any slice looks the same, but real data (see
        # src/m1_ingestion/real_scan.py) is genuinely non-uniform along
        # depth, and edge slices are often near-empty/out-of-focus.
        probe_path = corrected_path if corrected_path.exists() else raw_path
        with h5py.File(probe_path, "r") as f:
            slice_idx = f["volume"].shape[0] // 2

    # M1: raw vs corrected
    if raw_path and corrected_path.exists():
        with h5py.File(raw_path, "r") as f:
            raw_slice = f["volume"][slice_idx]
        with h5py.File(corrected_path, "r") as f:
            corrected_slice = f["volume"][slice_idx]
        dest = out_dir / f"{stem}_preview_01_m1_before_after.png"
        save_side_by_side(raw_slice, corrected_slice, str(dest), size=PREVIEW_SIZE)
        written.append(dest)
        print(f"M1 preview -> {dest}")

    # M2: corrected slice vs colourised segment labels
    if corrected_path.exists() and segmented_path.exists():
        with h5py.File(corrected_path, "r") as f:
            vol_slice = f["volume"][slice_idx]
        with h5py.File(segmented_path, "r") as f:
            label_slice = f["labels"][slice_idx]
        dest = out_dir / f"{stem}_preview_02_m2_segmentation.png"
        save_side_by_side(vol_slice, label_slice, str(dest), size=PREVIEW_SIZE, right_is_labels=True)
        written.append(dest)
        print(f"M2 preview -> {dest}")

    # M3/M4: flattened page (+ ink heatmap if available)
    if flattened_path.exists():
        layer = _pick_demo_layer(flattened_path, ink_path if ink_path.exists() else None)
        with h5py.File(flattened_path, "r") as f:
            page = f[f"layer_{layer}"]["page"][...]

        if ink_path.exists():
            with h5py.File(ink_path, "r") as f:
                ink_map = f[f"layer_{layer}"]["ink_probability"][...]
            dest = out_dir / f"{stem}_preview_03_m3m4_page_and_ink.png"
            save_side_by_side(page, ink_map, str(dest), size=PREVIEW_SIZE, right_cmap="hot")
            print(f"M3+M4 preview (layer {layer}) -> {dest}")
        else:
            dest = out_dir / f"{stem}_preview_03_m3_flattened_page.png"
            to_display_image(page, size=PREVIEW_SIZE).save(dest)
            print(f"M3 preview (layer {layer}) -> {dest}")
        written.append(dest)

    if not written:
        print(f"No pipeline output found for stem '{stem}' under {out_dir} — run the pipeline first.")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", required=True, help="Input file stem, e.g. 'phantom_001' or 'cells3d'")
    parser.add_argument("--out", required=True, help="Directory containing the pipeline's HDF5 outputs")
    parser.add_argument("--slice", type=int, default=None, help="Depth slice index for the M1/M2 previews (default: middle of the stack)")
    parser.add_argument("--raw", default=None, help="Explicit path to the original input HDF5 (auto-detected if omitted)")
    args = parser.parse_args()

    export_previews(args.stem, args.out, slice_idx=args.slice, raw_path=args.raw)


if __name__ == "__main__":
    main()
