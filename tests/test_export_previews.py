"""Tests for scripts/export_previews.py.

Builds a small end-to-end fixture (M1-M4 outputs, in the same directory
layout run_end_to_end.py produces) and checks the exporter finds the right
files and writes real, non-trivial PNGs from them — including the path-
resolution fix (the input file lives in data/phantoms/, not next to the
processed outputs in an --out subdirectory, so naive "look next door"
logic misses it).
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_previews import export_previews
from src.m1_ingestion.generate_phantom import generate_phantom_full, save_phantom
from src.m1_ingestion.preprocessing import correct_chunk
from src.m2_segmentation.segment import segment_volume
from src.m3_flattening.flatten import flatten_layer

CFG = {
    "seed": 0,
    "shape": [8, 64, 64],
    "n_layers": 4,
    "substrate_density": 0.3,
    "ink_density": 0.9,
    "beam_hardening_strength": 0.25,
    "ring_artifact_strength": 0.08,
    "n_rings": 3,
    "noise_sigma": 0.01,
    "chunk_depth": 4,
}


@pytest.fixture
def pipeline_outputs(tmp_path):
    """Lay out M1-M3 outputs the way run_end_to_end.py does: raw input
    under <root>/data/phantoms/, everything else under an --out directory
    that is NOT a sibling of data/phantoms."""
    raw, ground_truth, layer_id, ink_mask = generate_phantom_full(CFG)
    corrected = correct_chunk(raw)
    labels = segment_volume(corrected)

    phantoms_dir = tmp_path / "data" / "phantoms"
    phantoms_dir.mkdir(parents=True)
    save_phantom(phantoms_dir / "phantom_001.h5", raw, ground_truth, CFG, layer_id=layer_id, ink_mask=ink_mask)

    out_dir = tmp_path / "data" / "processed" / "e2e_run"
    out_dir.mkdir(parents=True)

    with h5py.File(out_dir / "phantom_001_corrected.h5", "w") as f:
        f.create_dataset("volume", data=corrected)
    with h5py.File(out_dir / "phantom_001_segmented.h5", "w") as f:
        f.create_dataset("labels", data=labels)

    with h5py.File(out_dir / "phantom_001_flattened.h5", "w") as f:
        for lid in range(1, int(labels.max()) + 1):
            result = flatten_layer(corrected, labels, lid, n_theta=60, arap_iters=2)
            grp = f.create_group(f"layer_{lid}")
            grp.create_dataset("page", data=result["page"])
            grp.create_dataset("coord_map", data=result["coord_map"])

    return tmp_path, out_dir


class TestExportPreviews:
    def test_finds_raw_input_across_directories(self, pipeline_outputs, monkeypatch):
        """This is the bug fix under test: out_dir.parent is
        <root>/data/processed, NOT <root>/data — the raw phantom is two
        levels away from out_dir, not a sibling of it."""
        root, out_dir = pipeline_outputs
        monkeypatch.chdir(root)

        written = export_previews("phantom_001", out_dir)
        names = [p.name for p in written]
        assert any("m1_before_after" in n for n in names)

    def test_m1_preview_is_nontrivial_image(self, pipeline_outputs, monkeypatch):
        root, out_dir = pipeline_outputs
        monkeypatch.chdir(root)

        export_previews("phantom_001", out_dir)
        m1_path = out_dir / "phantom_001_preview_01_m1_before_after.png"
        assert m1_path.exists()

        img = np.array(Image.open(m1_path))
        assert img.shape[0] > 0 and img.shape[1] > 0
        assert len(np.unique(img)) > 2  # not a blank/solid-colour image

    def test_m2_preview_shows_multiple_segment_colours(self, pipeline_outputs, monkeypatch):
        root, out_dir = pipeline_outputs
        monkeypatch.chdir(root)

        export_previews("phantom_001", out_dir)
        m2_path = out_dir / "phantom_001_preview_02_m2_segmentation.png"
        assert m2_path.exists()

        img = np.array(Image.open(m2_path).convert("RGB"))
        right_half = img[:, img.shape[1] // 2 :]
        distinct_colours = {tuple(c) for c in right_half.reshape(-1, 3)}
        assert len(distinct_colours) >= 2  # background + at least one segment colour

    def test_m3_preview_written_without_ink_map(self, pipeline_outputs, monkeypatch):
        """No M4 output in this fixture — exporter should still produce the
        M3-only flattened-page preview instead of erroring."""
        root, out_dir = pipeline_outputs
        monkeypatch.chdir(root)

        written = export_previews("phantom_001", out_dir)
        names = [p.name for p in written]
        assert any("m3_flattened_page" in n for n in names)
        assert not any("ink" in n for n in names)

    def test_explicit_raw_path_overrides_autodetect(self, pipeline_outputs, monkeypatch, tmp_path):
        root, out_dir = pipeline_outputs
        monkeypatch.chdir(root)

        raw, ground_truth, layer_id, ink_mask = generate_phantom_full(CFG)
        alt_path = tmp_path / "alt_phantom.h5"
        save_phantom(alt_path, raw, ground_truth, CFG, layer_id=layer_id, ink_mask=ink_mask)

        written = export_previews("phantom_001", out_dir, raw_path=alt_path)
        assert any("m1_before_after" in p.name for p in written)

    def test_middle_slice_used_by_default(self, pipeline_outputs, monkeypatch):
        """slice_idx=None should resolve to the middle of the stack, not 0."""
        root, out_dir = pipeline_outputs
        monkeypatch.chdir(root)

        with h5py.File(out_dir / "phantom_001_corrected.h5", "r") as f:
            expected_middle = f["volume"].shape[0] // 2

        called_with = {}
        import scripts.export_previews as ep

        original = ep.save_side_by_side

        def spy(left, right, *args, **kwargs):
            called_with["left"] = left
            return original(left, right, *args, **kwargs)

        monkeypatch.setattr(ep, "save_side_by_side", spy)
        export_previews("phantom_001", out_dir)

        with h5py.File(out_dir / "phantom_001_corrected.h5", "r") as f:
            expected_slice = f["volume"][expected_middle]
        np.testing.assert_array_equal(called_with["left"], expected_slice)
