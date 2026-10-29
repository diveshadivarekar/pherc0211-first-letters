"""Official-tool ink detection for tifxyz pages: VC3D ``vc_render_tifxyz`` -> villa ``vesuvius`` ``ink_9um`` inference -> page ``.h5``.

Replaces M4's small U-Net on real scrolls. The checkpoint is trained on ~9 um data, so pages are rendered at ``--scale`` 0.267 for a 2.4 um volume
(``--slice-step`` 4 voxels). Writes ``<out>/<layer>.h5`` (forward) and ``<out>/<layer>_reverse.h5`` in the format ``eval_ink_labels.py`` reads.

Usage:
    python scripts/run_ink9um.py --tifxyz-dir data/processed/tifxyz/paris4_split --layers layer_1 layer_13 --out data/processed/paris4/ink9um \
        --remote-url s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr \
        --offset 62080 17280 18816 --checkpoint ../ink9um/hybrid_3d2d-seed42/step-075000.pth --python ../inkenv/Scripts/python.exe
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ink_from_vc3d import build  # noqa: E402

VC_RENDER = Path("C:/Program Files/VC3D/bin/vc_render_tifxyz.exe")


def run_layer(layer: Path, out: Path, a: argparse.Namespace) -> None:
    name = layer.name
    render = out / f"{name}_render.zarr"
    if not render.exists():
        subprocess.run([str(a.vc_render), "--volume", str(a.volume_cache), "--remote-url", a.remote_url, "--group-idx", "0", "--scale", str(a.scale),
                        "--segmentation", str(layer), "--num-slices", str(a.num_slices), "--slice-step", str(a.slice_step), "--cache-gb", "8",
                        "--zarr-output", str(render), "--timeout", "30"], check=True)
    tif = out / f"{name}.tif"
    if not tif.exists():
        subprocess.run([str(a.python), "-m", "vesuvius.ink_detection.inference.infer", str(render), str(a.checkpoint), str(tif),
                        "--overlap", "0.5", "--blend-mode", "hann", "--batch-size", "4", "--direction", "both"], check=True)
    for suffix in ("", "_reverse"):
        build(layer, out / f"{name}{suffix}.tif", out / f"{name}{suffix}.h5", name, "w", a.offset, a.scale)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tifxyz-dir", type=Path, required=True)
    p.add_argument("--layers", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--remote-url", required=True)
    p.add_argument("--volume-cache", type=Path, default=Path("../volume-cache/volume.zarr"))
    p.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("Z", "Y", "X"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True, help="interpreter of an env with the villa `vesuvius` package + CUDA torch")
    p.add_argument("--vc-render", type=Path, default=VC_RENDER)
    p.add_argument("--scale", type=float, default=0.267)
    p.add_argument("--slice-step", type=int, default=4)
    p.add_argument("--num-slices", type=int, default=28)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for L in a.layers:
        run_layer(a.tifxyz_dir / L, a.out, a)
        print(f"done {L}")
